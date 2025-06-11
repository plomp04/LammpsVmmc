// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors:
    Lester Hedges (Bristol, UK)
    Ford Cadman, Oliver Henrich (University of Strathclyde, Glasgow)
------------------------------------------------------------------------- */

#include "fix_vmmc.h"
#include "VMMC.h"

#include "angle.h"
#include "atom.h"
#include "atom_vec.h"
#include "bond.h"
#include "comm.h"
#include "compute.h"
#include "dihedral.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "group.h"
#include "improper.h"
#include "kspace.h"
#include "math_const.h"
#include "math_extra.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "pair.h"
#include "random_park.h"
#include "region.h"
#include "update.h"

#include <cmath>
#include <cstring>
#include <exception>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

// large energy value used to signal overlap

static constexpr double MAXENERGYSIGNAL = 1.0e100;

// this must be lower than MAXENERGYSIGNAL
// by a large amount, so that it is still
// less than total energy when negative
// energy contributions are added to MAXENERGYSIGNAL

static constexpr double MAXENERGYTEST = 1.0e50;

enum { EXCHATOM, EXCHMOL };          // exchmode
enum { NONE, MOVEATOM, MOVEMOL };    // movemode

/* ---------------------------------------------------------------------- */

FixVMMC::FixVMMC(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), region(nullptr), idregion(nullptr), full_flag(false),
    groupstrings(nullptr), grouptypestrings(nullptr), grouptypebits(nullptr), grouptypes(nullptr),
    local_gas_list(nullptr), random_equal(nullptr), random_unequal(nullptr)
{
  if (narg < 11) utils::missing_cmd_args(FLERR, "fix vmmc", error);

  if (atom->molecular == Atom::TEMPLATE)
    error->all(FLERR,"Fix vmmc does not (yet) work with atom_style template");

  dynamic_group_allow = 1;

  vector_flag = 1;
  size_vector = 8;
  global_freq = 1;
  extvector = 0;
  restart_global = 1;
  time_depend = 1;

  ngroups = 0;
  ngrouptypes = 0;
  triclinic = domain->triclinic;

  // required args

  nevery = utils::inumeric(FLERR, arg[3], false, lmp);
  nexchanges = utils::inumeric(FLERR, arg[4], false, lmp);
  nmcmoves = utils::inumeric(FLERR, arg[5], false, lmp);
  nvmmc_type = utils::expand_type_int(FLERR, arg[6], Atom::ATOM, lmp);
  seed = utils::inumeric(FLERR, arg[7], false, lmp);
  reservoir_temperature = utils::numeric(FLERR, arg[8], false, lmp);
  chemical_potential = utils::numeric(FLERR, arg[9], false, lmp);
  displace = utils::numeric(FLERR, arg[10], false, lmp);

  if (nevery <= 0) error->all(FLERR, "Illegal fix vmmc command");
  if (nexchanges < 0) error->all(FLERR, "Illegal fix vmmc command");
  if (nmcmoves < 0) error->all(FLERR, "Illegal fix vmmc command");
  if (seed <= 0) error->all(FLERR, "Illegal fix vmmc command");
  if (reservoir_temperature < 0.0)
    error->all(FLERR, "Illegal fix vmmc command");
  if (displace < 0.0) error->all(FLERR, "Illegal fix vmmc command");

  // read options from end of input line

  options(narg-11,&arg[11]);

  // random number generator, same for all procs

  random_equal = new RanPark(lmp,seed);

  // random number generator, not the same for all procs

  random_unequal = new RanPark(lmp,seed);

  // error checks on region and its extent being inside simulation box

  region_xlo = region_xhi = region_ylo = region_yhi = region_zlo = region_zhi = 0.0;
  if (region) {
    if (region->bboxflag == 0)
      error->all(FLERR,"Fix vmmc region does not support a bounding box");
    if (region->dynamic_check())
      error->all(FLERR,"Fix vmmc region cannot be dynamic");

    region_xlo = region->extent_xlo;
    region_xhi = region->extent_xhi;
    region_ylo = region->extent_ylo;
    region_yhi = region->extent_yhi;
    region_zlo = region->extent_zlo;
    region_zhi = region->extent_zhi;

    // estimate region volume using MC trials

    double coord[3];
    int inside = 0;
    int attempts = 10000000;
    for (int i = 0; i < attempts; i++) {
      coord[0] = region_xlo + random_equal->uniform() * (region_xhi-region_xlo);
      coord[1] = region_ylo + random_equal->uniform() * (region_yhi-region_ylo);
      coord[2] = region_zlo + random_equal->uniform() * (region_zhi-region_zlo);
      if (region->match(coord[0],coord[1],coord[2]) != 0)
        inside++;
    }

    double max_region_volume = (region_xhi - region_xlo) *
      (region_yhi - region_ylo) * (region_zhi - region_zlo);

    region_volume = max_region_volume * static_cast<double>(inside) / static_cast<double>(attempts);
  }

  if (charge_flag && atom->q == nullptr)
    error->all(FLERR,"Fix vmmc atom has charge, but atom style does not");

  // compute the number of MC cycles that occur nevery timesteps

  ncycles = nexchanges + nmcmoves;

  // set up reneighboring

  force_reneighbor = 1;
  next_reneighbor = update->ntimestep + 1;

  // zero out counters

  mc_active = 0;

  ntranslation_attempts = 0.0;
  ntranslation_successes = 0.0;
  nrotation_attempts = 0.0;
  nrotation_successes = 0.0;

  vmmc_nmax = 0;
  local_gas_list = nullptr;
}

/* ----------------------------------------------------------------------
   parse optional parameters at end of input line
------------------------------------------------------------------------- */

void FixVMMC::options(int narg, char **arg)
{
  if (narg < 0) error->all(FLERR,"Illegal fix vmmc command");

  // defaults

  exchmode = EXCHATOM;
  movemode = NONE;
  patomtrans = 0.0;
  pmctot = 0.0;
  max_rotation_angle = 10*MY_PI/180;
  region_volume = 0;
  max_region_attempts = 1000;
  molecule_group = 0;
  molecule_group_bit = 0;
  molecule_group_inversebit = 0;
  exclusion_group = 0;
  exclusion_group_bit = 0;
  pressure_flag = false;
  pressure = 0.0;
  fugacity_coeff = 1.0;
  charge = 0.0;
  charge_flag = false;
  full_flag = false;
  ngroups = 0;
  int ngroupsmax = 0;
  groupstrings = nullptr;
  ngrouptypes = 0;
  int ngrouptypesmax = 0;
  grouptypestrings = nullptr;
  grouptypes = nullptr;
  grouptypebits = nullptr;
  energy_intra = 0.0;
  tfac_insert = 1.0;
  overlap_cutoffsq = 0.0;
  overlap_flag = 0;
  min_ngas = -1;
  max_ngas = INT_MAX;

  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"mcmoves") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix vmmc command");
      patomtrans = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      if (patomtrans < 0)
        error->all(FLERR,"Illegal fix vmmc command");
      pmctot = patomtrans;
      if (pmctot <= 0)
        error->all(FLERR,"Illegal fix vmmc command");
      iarg += 4;
    } else if (strcmp(arg[iarg],"region") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix vmmc command");
      region = domain->get_region_by_id(arg[iarg+1]);
      if (!region) error->all(FLERR,"Region {} for fix vmmc does not exist",arg[iarg+1]);
      idregion = utils::strdup(arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"maxangle") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix vmmc command");
      max_rotation_angle = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      max_rotation_angle *= MY_PI/180;
      iarg += 2;
    } else if (strcmp(arg[iarg],"pressure") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix vmmc command");
      pressure = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      pressure_flag = true;
      iarg += 2;
    } else if (strcmp(arg[iarg],"fugacity_coeff") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix vmmc command");
      fugacity_coeff = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"charge") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix vmmc command");
      charge = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      charge_flag = true;
      iarg += 2;
    }  else if (strcmp(arg[iarg],"full_energy") == 0) {
      full_flag = true;
      iarg += 1;
    } else if (strcmp(arg[iarg],"group") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix vmmc command");
      if (ngroups >= ngroupsmax) {
        ngroupsmax = ngroups+1;
        groupstrings = (char **)
          memory->srealloc(groupstrings,
                           ngroupsmax*sizeof(char *),
                           "fix_vmmc:groupstrings");
      }
      groupstrings[ngroups] = utils::strdup(arg[iarg+1]);
      ngroups++;
      iarg += 2;
    } else if (strcmp(arg[iarg],"grouptype") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix vmmc command");
      if (ngrouptypes >= ngrouptypesmax) {
        ngrouptypesmax = ngrouptypes+1;
        grouptypes = (int*) memory->srealloc(grouptypes,ngrouptypesmax*sizeof(int),
                         "fix_vmmc:grouptypes");
        grouptypestrings = (char**)
          memory->srealloc(grouptypestrings,
                           ngrouptypesmax*sizeof(char *),
                           "fix_vmmc:grouptypestrings");
      }
      grouptypes[ngrouptypes] = utils::expand_type_int(FLERR, arg[iarg+1], Atom::ATOM, lmp);
      grouptypestrings[ngrouptypes] = utils::strdup(arg[iarg+2]);
      ngrouptypes++;
      iarg += 3;
    } else if (strcmp(arg[iarg],"intra_energy") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix vmmc command");
      energy_intra = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"tfac_insert") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix vmmc command");
      tfac_insert = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"overlap_cutoff") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix vmmc command");
      double rtmp = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      overlap_cutoffsq = rtmp*rtmp;
      overlap_flag = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg],"min") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix vmmc command");
      min_ngas = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"max") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix vmmc command");
      max_ngas = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else error->all(FLERR,"Illegal fix vmmc command");
  }
}

/* ---------------------------------------------------------------------- */

FixVMMC::~FixVMMC()
{
  delete[] idregion;
  delete random_equal;
  delete random_unequal;

  memory->destroy(local_gas_list);

  if (ngroups > 0) {
    for (int igroup = 0; igroup < ngroups; igroup++)
      delete[] groupstrings[igroup];
    memory->sfree(groupstrings);
  }

  if (ngrouptypes > 0) {
    memory->destroy(grouptypes);
    memory->destroy(grouptypebits);
    for (int igroup = 0; igroup < ngrouptypes; igroup++)
      delete[] grouptypestrings[igroup];
    memory->sfree(grouptypestrings);
  }

  // delete exclusion group created in init()
  // delete molecule group created in init()
  // unset neighbor exclusion settings made in init()
  // not necessary if group and neighbor classes already destroyed
  //   when LAMMPS exits

  if (exclusion_group_bit && group) {
    auto group_id = std::string("FixVMMC:vmmc_exclusion_group:") + id;
    try {
      group->assign(group_id + " delete");
    } catch (std::exception &e) {
      if (comm->me == 0)
        fprintf(stderr, "Error deleting group %s: %s\n", group_id.c_str(), e.what());
    }
  }

  if (molecule_group_bit && group) {
    auto group_id = std::string("FixVMMC:rotation_gas_atoms:") + id;
    try {
      group->assign(group_id + " delete");
    } catch (std::exception &e) {
      if (comm->me == 0)
        fprintf(stderr, "Error deleting group %s: %s\n", group_id.c_str(), e.what());
    }
  }

  if (full_flag && group && neighbor) {
    int igroupall = group->find("all");
    neighbor->exclusion_group_group_delete(exclusion_group,igroupall);
  }
}

/* ---------------------------------------------------------------------- */

int FixVMMC::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixVMMC::init()
{
  if (!atom->mass) error->all(FLERR, "Fix vmmc requires per atom type masses");
  if (atom->rmass_flag && (comm->me == 0))
    error->warning(FLERR, "Fix vmmc will use per atom type masses for velocity initialization");

  triclinic = domain->triclinic;

  // set index and check validity of region

  if (idregion) {
    region = domain->get_region_by_id(idregion);
    if (!region) error->all(FLERR, "Region {} for fix vmmc does not exist", idregion);
  }

  if (region) {
    if (region->bboxflag == 0)
      error->all(FLERR,"Fix vmmc region does not support a bounding box");
    if (region->dynamic_check())
      error->all(FLERR,"Fix vmmc region cannot be dynamic");

    region_xlo = region->extent_xlo;
    region_xhi = region->extent_xhi;
    region_ylo = region->extent_ylo;
    region_yhi = region->extent_yhi;
    region_zlo = region->extent_zlo;
    region_zhi = region->extent_zhi;

    if (triclinic) {
      if ((region_xlo < domain->boxlo_bound[0]) || (region_xhi > domain->boxhi_bound[0]) ||
          (region_ylo < domain->boxlo_bound[1]) || (region_yhi > domain->boxhi_bound[1]) ||
          (region_zlo < domain->boxlo_bound[2]) || (region_zhi > domain->boxhi_bound[2])) {
        error->all(FLERR,"Fix vmmc region extends outside simulation box");
      }
    } else {
      if ((region_xlo < domain->boxlo[0]) || (region_xhi > domain->boxhi[0]) ||
          (region_ylo < domain->boxlo[1]) || (region_yhi > domain->boxhi[1]) ||
          (region_zlo < domain->boxlo[2]) || (region_zhi > domain->boxhi[2]))
        error->all(FLERR,"Fix vmmc region extends outside simulation box");
    }
  }

  // set probabilities for MC moves

  if (nmcmoves > 0) {
    if (pmctot == 0.0) {
      if (exchmode == EXCHATOM) {
        movemode = MOVEATOM;
        patomtrans = 1.0;
      }
    }
    else {
      movemode = MOVEATOM;
      patomtrans /= pmctot;
    }
  } else movemode = NONE;

  // decide whether to switch to the full_energy option

  if (!full_flag) {
    if ((force->kspace) ||
        (force->pair == nullptr) ||
        (force->pair->single_enable == 0) ||
        (force->pair_match("^hybrid",0)) ||
        (force->pair_match("^eam",0)) ||
        (force->pair->tail_flag)) {
      full_flag = true;
      if (comm->me == 0)
        error->warning(FLERR,"Fix vmmc using full_energy option");
    }
  }

  if (full_flag) c_pe = modify->compute[modify->find_compute("thermo_pe")];

  int *type = atom->type;

  if (exchmode == EXCHATOM) {
    if (nvmmc_type <= 0 || nvmmc_type > atom->ntypes)
      error->all(FLERR,"Invalid atom type in fix vmmc command");
  }

  // if atoms are exchanged, warn if any deletable atom has a mol ID

  if ((exchmode == EXCHATOM) && atom->molecule_flag) {
    tagint *molecule = atom->molecule;
    int flag = 0;
    for (int i = 0; i < atom->nlocal; i++)
      if (type[i] == nvmmc_type)
        if (molecule[i]) flag = 1;
    int flagall;
    MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);
    if (flagall)
      error->all(FLERR, "Fix vmmc cannot exchange individual atoms belonging to a molecule");
  }

  if (domain->dimension == 2)
    error->all(FLERR,"Cannot use fix vmmc in a 2d simulation");

  // create a new group for interaction exclusions
  // used for attempted atom or molecule deletions
  // skip if already exists from previous init()

  if (full_flag && !exclusion_group_bit) {

    // create unique group name for atoms to be excluded

    auto group_id = std::string("FixVMMC:vmmc_exclusion_group:") + id;
    group->assign(group_id + " subtract all all");
    exclusion_group = group->find(group_id);
    if (exclusion_group == -1)
      error->all(FLERR,"Could not find fix vmmc exclusion group ID");
    exclusion_group_bit = group->bitmask[exclusion_group];

    // neighbor list exclusion setup
    // turn off interactions between group all and the exclusion group

    neighbor->modify_params(fmt::format("exclude group {} all",group_id));
  }

  // get all of the needed molecule data if exchanging
  // or moving molecules, otherwise just get the gas mass

  gas_mass = atom->mass[nvmmc_type];

  if (gas_mass <= 0.0)
    error->all(FLERR,"Illegal fix vmmc gas mass <= 0");

  // check that no deletable atoms are in atom->firstgroup
  // deleting such an atom would not leave firstgroup atoms first

  if (atom->firstgroup >= 0) {
    int *mask = atom->mask;
    int firstgroupbit = group->bitmask[atom->firstgroup];

    int flag = 0;
    for (int i = 0; i < atom->nlocal; i++)
      if ((mask[i] == groupbit) && (mask[i] && firstgroupbit)) flag = 1;

    int flagall;
    MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);

    if (flagall)
      error->all(FLERR,"Cannot do VMMC on atoms in atom_modify first group");
  }

  // compute beta, lambda, sigma, and the zz factor
  // For LJ units, lambda=1
  beta = 1.0/(force->boltz*reservoir_temperature);
  if (strcmp(update->unit_style,"lj") == 0)
    zz = exp(beta*chemical_potential);
  else {
    double lambda = sqrt(force->hplanck*force->hplanck/
                         (2.0*MY_PI*gas_mass*force->mvv2e*
                        force->boltz*reservoir_temperature));
    zz = exp(beta*chemical_potential)/(pow(lambda,3.0));
  }

  sigma = sqrt(force->boltz*reservoir_temperature*tfac_insert/gas_mass/force->mvv2e);
  if (pressure_flag) zz = pressure*fugacity_coeff*beta/force->nktv2p;

  imagezero = ((imageint) IMGMAX << IMG2BITS) |
             ((imageint) IMGMAX << IMGBITS) | IMGMAX;

  // warning if group id is "all"

  if ((comm->me == 0) && (groupbit & 1))
    error->warning(FLERR, "Fix vmmc is being applied "
                   "to the default group all");

  // construct group bitmask for all new atoms
  // aggregated over all group keywords

  groupbitall = 1 | groupbit;
  for (int igroup = 0; igroup < ngroups; igroup++) {
    int jgroup = group->find(groupstrings[igroup]);
    if (jgroup == -1)
      error->all(FLERR,"Could not find specified fix vmmc group ID");
    groupbitall |= group->bitmask[jgroup];
  }

  // construct group type bitmasks
  // not aggregated over all group keywords

  if (ngrouptypes > 0) {
    memory->create(grouptypebits,ngrouptypes,"fix_vmmc:grouptypebits");
    for (int igroup = 0; igroup < ngrouptypes; igroup++) {
      int jgroup = group->find(grouptypestrings[igroup]);
      if (jgroup == -1)
        error->all(FLERR,"Could not find specified fix vmmc group ID");
      grouptypebits[igroup] = group->bitmask[jgroup];
    }
  }

}

/* ----------------------------------------------------------------------
   attempt Monte Carlo translations, rotations, insertions, and deletions
   done before exchange, borders, reneighbor
   so that ghost atoms and neighbor lists will be correct
------------------------------------------------------------------------- */

void FixVMMC::pre_exchange()
{
  // just return if should not be called on this timestep

  if (next_reneighbor != update->ntimestep) return;

  mc_active = 1;

  xlo = domain->boxlo[0];
  xhi = domain->boxhi[0];
  ylo = domain->boxlo[1];
  yhi = domain->boxhi[1];
  zlo = domain->boxlo[2];
  zhi = domain->boxhi[2];
  if (triclinic) {
    sublo = domain->sublo_lamda;
    subhi = domain->subhi_lamda;
  } else {
    sublo = domain->sublo;
    subhi = domain->subhi;
  }

  if (region) volume = region_volume;
  else volume = domain->xprd * domain->yprd * domain->zprd;

  if (triclinic) domain->x2lamda(atom->nlocal);
  domain->pbc();
  comm->exchange();
  atom->nghost = 0;
  comm->borders();
  if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
  update_gas_atoms_list();

  if (full_flag) {
    energy_stored = energy_full();
    if (overlap_flag && energy_stored > MAXENERGYTEST)
        error->warning(FLERR,"Energy of old configuration in "
                       "fix vmmc is > MAXENERGYTEST.");

    for (int i = 0; i < ncycles; i++) {
      int ixm = static_cast<int>(random_equal->uniform()*ncycles) + 1;
      if (ixm <= nmcmoves) {
        double xmcmove = random_equal->uniform();
        if (xmcmove < patomtrans) attempt_atomic_translation_full();
      }
    }
    if (triclinic) domain->x2lamda(atom->nlocal);
    domain->pbc();
    comm->exchange();
    atom->nghost = 0;
    comm->borders();
    if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);

  } else {

    for (int i = 0; i < ncycles; i++) {
      int ixm = static_cast<int>(random_equal->uniform()*ncycles) + 1;
      if (ixm <= nmcmoves) {
        double xmcmove = random_equal->uniform();
        if (xmcmove < patomtrans) attempt_atomic_translation();
      }
    }
  }

  next_reneighbor = update->ntimestep + nevery;

  mc_active = 0;
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

void FixVMMC::attempt_atomic_translation()
{
  ntranslation_attempts += 1.0;

  if (ngas == 0) return;

  int i = pick_random_gas_atom();

  int success = 0;
  if (i >= 0) {
    double **x = atom->x;
    double energy_before = energy(i,nvmmc_type,-1,x[i]);
    if (overlap_flag && energy_before > MAXENERGYTEST)
        error->warning(FLERR,"Energy of old configuration in fix vmmc is > MAXENERGYTEST.");
    double rsq = 1.1;
    double rx,ry,rz;
    rx = ry = rz = 0.0;
    double coord[3];
    while (rsq > 1.0) {
      rx = 2*random_unequal->uniform() - 1.0;
      ry = 2*random_unequal->uniform() - 1.0;
      rz = 2*random_unequal->uniform() - 1.0;
      rsq = rx*rx + ry*ry + rz*rz;
    }
    coord[0] = x[i][0] + displace*rx;
    coord[1] = x[i][1] + displace*ry;
    coord[2] = x[i][2] + displace*rz;
    if (region) {
      while (region->match(coord[0],coord[1],coord[2]) == 0) {
        rsq = 1.1;
        while (rsq > 1.0) {
          rx = 2*random_unequal->uniform() - 1.0;
          ry = 2*random_unequal->uniform() - 1.0;
          rz = 2*random_unequal->uniform() - 1.0;
          rsq = rx*rx + ry*ry + rz*rz;
        }
        coord[0] = x[i][0] + displace*rx;
        coord[1] = x[i][1] + displace*ry;
        coord[2] = x[i][2] + displace*rz;
      }
    }
    if (!domain->inside_nonperiodic(coord))
      error->one(FLERR,"Fix vmmc put atom outside box");

    double energy_after = energy(i,nvmmc_type,-1,coord);

    if (energy_after < MAXENERGYTEST &&
        random_unequal->uniform() <
        exp(beta*(energy_before - energy_after))) {
      x[i][0] = coord[0];
      x[i][1] = coord[1];
      x[i][2] = coord[2];
      success = 1;
    }
  }

  int success_all = 0;
  MPI_Allreduce(&success,&success_all,1,MPI_INT,MPI_MAX,world);

  if (success_all) {
    if (triclinic) domain->x2lamda(atom->nlocal);
    domain->pbc();
    comm->exchange();
    atom->nghost = 0;
    comm->borders();
    if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
    update_gas_atoms_list();
    ntranslation_successes += 1.0;
  }
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

void FixVMMC::attempt_atomic_translation_full()
{
  ntranslation_attempts += 1.0;

  if (ngas == 0) return;

  double energy_before = energy_stored;

  int i = pick_random_gas_atom();

  double **x = atom->x;
  double xtmp[3];

  xtmp[0] = xtmp[1] = xtmp[2] = 0.0;

  tagint tmptag = -1;

  if (i >= 0) {

    double rsq = 1.1;
    double rx,ry,rz;
    rx = ry = rz = 0.0;
    double coord[3];
    while (rsq > 1.0) {
      rx = 2*random_unequal->uniform() - 1.0;
      ry = 2*random_unequal->uniform() - 1.0;
      rz = 2*random_unequal->uniform() - 1.0;
      rsq = rx*rx + ry*ry + rz*rz;
    }
    coord[0] = x[i][0] + displace*rx;
    coord[1] = x[i][1] + displace*ry;
    coord[2] = x[i][2] + displace*rz;
    if (region) {
      while (region->match(coord[0],coord[1],coord[2]) == 0) {
        rsq = 1.1;
        while (rsq > 1.0) {
          rx = 2*random_unequal->uniform() - 1.0;
          ry = 2*random_unequal->uniform() - 1.0;
          rz = 2*random_unequal->uniform() - 1.0;
          rsq = rx*rx + ry*ry + rz*rz;
        }
        coord[0] = x[i][0] + displace*rx;
        coord[1] = x[i][1] + displace*ry;
        coord[2] = x[i][2] + displace*rz;
      }
    }
    if (!domain->inside_nonperiodic(coord))
      error->one(FLERR,"Fix vmmc put atom outside box");
    xtmp[0] = x[i][0];
    xtmp[1] = x[i][1];
    xtmp[2] = x[i][2];
    x[i][0] = coord[0];
    x[i][1] = coord[1];
    x[i][2] = coord[2];

    tmptag = atom->tag[i];
  }

  double energy_after = energy_full();

  if (energy_after < MAXENERGYTEST &&
      random_equal->uniform() <
      exp(beta*(energy_before - energy_after))) {
    energy_stored = energy_after;
    ntranslation_successes += 1.0;
  } else {

    tagint tmptag_all;
    MPI_Allreduce(&tmptag,&tmptag_all,1,MPI_LMP_TAGINT,MPI_MAX,world);

    double xtmp_all[3];
    MPI_Allreduce(&xtmp,&xtmp_all,3,MPI_DOUBLE,MPI_SUM,world);

    for (int i = 0; i < atom->nlocal; i++) {
      if (tmptag_all == atom->tag[i]) {
        x[i][0] = xtmp_all[0];
        x[i][1] = xtmp_all[1];
        x[i][2] = xtmp_all[2];
      }
    }
    energy_stored = energy_before;
  }
  update_gas_atoms_list();
}

/* ----------------------------------------------------------------------
   compute particle's interaction energy with the rest of the system by
   looping over all atoms in the sub-domain including ghosts.
------------------------------------------------------------------------- */

double FixVMMC::energy(int i, int itype, tagint imolecule, double *coord)
{
  double delx,dely,delz,rsq;

  double **x = atom->x;
  int *type = atom->type;
  int nall = atom->nlocal + atom->nghost;
  pair = force->pair;
  cutsq = force->pair->cutsq;

  double fpair = 0.0;
  double factor_coul = 1.0;
  double factor_lj = 1.0;

  double total_energy = 0.0;

  for (int j = 0; j < nall; j++) {

    if (i == j) continue;

    delx = coord[0] - x[j][0];
    dely = coord[1] - x[j][1];
    delz = coord[2] - x[j][2];
    rsq = delx*delx + dely*dely + delz*delz;
    int jtype = type[j];

    // if overlap check requested, if overlap,
    // return signal value for energy

    if (overlap_flag && rsq < overlap_cutoffsq)
      return MAXENERGYSIGNAL;

    if (rsq < cutsq[itype][jtype])
      total_energy +=
        pair->single(i,j,itype,jtype,rsq,factor_coul,factor_lj,fpair);
  }

  return total_energy;
}

/* ----------------------------------------------------------------------
   compute system potential energy
------------------------------------------------------------------------- */

double FixVMMC::energy_full()
{

  if (triclinic) domain->x2lamda(atom->nlocal);
  domain->pbc();
  comm->exchange();
  atom->nghost = 0;
  comm->borders();
  if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
  if (modify->n_pre_neighbor) modify->pre_neighbor();
  neighbor->build(1);
  int eflag = 1;
  int vflag = 0;

  // if overlap check requested, if overlap,
  // return signal value for energy

  if (overlap_flag) {
    int overlaptestall;
    int overlaptest = 0;
    double delx,dely,delz,rsq;
    double **x = atom->x;
    int nall = atom->nlocal + atom->nghost;
    for (int i = 0; i < atom->nlocal; i++) {
      for (int j = i+1; j < nall; j++) {

        delx = x[i][0] - x[j][0];
        dely = x[i][1] - x[j][1];
        delz = x[i][2] - x[j][2];
        rsq = delx*delx + dely*dely + delz*delz;

        if (rsq < overlap_cutoffsq) {
          overlaptest = 1;
          break;
        }
      }
      if (overlaptest) break;
    }
    MPI_Allreduce(&overlaptest, &overlaptestall, 1, MPI_INT, MPI_MAX, world);
    if (overlaptestall) return MAXENERGYSIGNAL;
  }

  // clear forces so they don't accumulate over multiple
  // calls within fix vmmc timestep, e.g. for fix shake

  size_t nbytes = sizeof(double) * (atom->nlocal + atom->nghost);
  if (nbytes) memset(&atom->f[0][0],0,3*nbytes);

  if (modify->n_pre_force) modify->pre_force(vflag);

  if (force->pair) force->pair->compute(eflag,vflag);

  if (atom->molecular != Atom::ATOMIC) {
    if (force->bond) force->bond->compute(eflag,vflag);
    if (force->angle) force->angle->compute(eflag,vflag);
    if (force->dihedral) force->dihedral->compute(eflag,vflag);
    if (force->improper) force->improper->compute(eflag,vflag);
  }

  if (force->kspace) force->kspace->compute(eflag,vflag);

  if (modify->n_post_force_any) modify->post_force(vflag);

  // NOTE: all fixes with energy_global_flag set and which
  //   operate at pre_force() or post_force()
  //   and which user has enabled via fix_modify energy yes,
  //   will contribute to total MC energy via pe->compute_scalar()

  update->eflag_global = update->ntimestep;
  double total_energy = c_pe->compute_scalar();

  return total_energy;
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

int FixVMMC::pick_random_gas_atom()
{
  int i = -1;
  int iwhichglobal = static_cast<int> (ngas*random_equal->uniform());
  if ((iwhichglobal >= ngas_before) &&
      (iwhichglobal < ngas_before + ngas_local)) {
    int iwhichlocal = iwhichglobal - ngas_before;
    i = local_gas_list[iwhichlocal];
  }

  return i;
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

tagint FixVMMC::pick_random_gas_molecule()
{
  int iwhichglobal = static_cast<int> (ngas*random_equal->uniform());
  tagint gas_molecule_id = 0;
  if ((iwhichglobal >= ngas_before) &&
      (iwhichglobal < ngas_before + ngas_local)) {
    int iwhichlocal = iwhichglobal - ngas_before;
    int i = local_gas_list[iwhichlocal];
    gas_molecule_id = atom->molecule[i];
  }

  tagint gas_molecule_id_all = 0;
  MPI_Allreduce(&gas_molecule_id,&gas_molecule_id_all,1,
                MPI_LMP_TAGINT,MPI_MAX,world);

  return gas_molecule_id_all;
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

void FixVMMC::toggle_intramolecular(int i)
{
  if (atom->avec->bonds_allow)
    for (int m = 0; m < atom->num_bond[i]; m++)
      atom->bond_type[i][m] = -atom->bond_type[i][m];

  if (atom->avec->angles_allow)
    for (int m = 0; m < atom->num_angle[i]; m++)
      atom->angle_type[i][m] = -atom->angle_type[i][m];

  if (atom->avec->dihedrals_allow)
    for (int m = 0; m < atom->num_dihedral[i]; m++)
      atom->dihedral_type[i][m] = -atom->dihedral_type[i][m];

  if (atom->avec->impropers_allow)
    for (int m = 0; m < atom->num_improper[i]; m++)
      atom->improper_type[i][m] = -atom->improper_type[i][m];
}

/* ----------------------------------------------------------------------
   update the list of gas atoms
------------------------------------------------------------------------- */

void FixVMMC::update_gas_atoms_list()
{
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  double **x = atom->x;

  if (atom->nmax > vmmc_nmax) {
    memory->sfree(local_gas_list);
    vmmc_nmax = atom->nmax;
    local_gas_list = (int *) memory->smalloc(vmmc_nmax*sizeof(int),
     "VMMC:local_gas_list");
  }

  ngas_local = 0;

  if (region) {

    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        if (region->match(x[i][0],x[i][1],x[i][2]) == 1) {
          local_gas_list[ngas_local] = i;
          ngas_local++;
        }
      }
    }

  } else {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        local_gas_list[ngas_local] = i;
        ngas_local++;
      }
    }
  }

  MPI_Allreduce(&ngas_local,&ngas,1,MPI_INT,MPI_SUM,world);
  MPI_Scan(&ngas_local,&ngas_before,1,MPI_INT,MPI_SUM,world);
  ngas_before -= ngas_local;
}

/* ----------------------------------------------------------------------
  return acceptance ratios
------------------------------------------------------------------------- */

double FixVMMC::compute_vector(int n)
{
  if (n == 0) return ntranslation_attempts;
  if (n == 1) return ntranslation_successes;
  if (n == 6) return nrotation_attempts;
  if (n == 7) return nrotation_successes;
  return 0.0;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixVMMC::memory_usage()
{
  double bytes = (double)vmmc_nmax * sizeof(int);
  return bytes;
}

/* ----------------------------------------------------------------------
   pack entire state of Fix into one write
------------------------------------------------------------------------- */

void FixVMMC::write_restart(FILE *fp)
{
  int n = 0;
  double list[12];
  list[n++] = random_equal->state();
  list[n++] = random_unequal->state();
  list[n++] = ubuf(next_reneighbor).d;
  list[n++] = ntranslation_attempts;
  list[n++] = ntranslation_successes;
  list[n++] = nrotation_attempts;
  list[n++] = nrotation_successes;
  list[n++] = ubuf(update->ntimestep).d;

  if (comm->me == 0) {
    int size = n * sizeof(double);
    fwrite(&size,sizeof(int),1,fp);
    fwrite(list,sizeof(double),n,fp);
  }
}

/* ----------------------------------------------------------------------
   use state info from restart file to restart the Fix
------------------------------------------------------------------------- */

void FixVMMC::restart(char *buf)
{
  int n = 0;
  auto list = (double *) buf;

  seed = static_cast<int> (list[n++]);
  random_equal->reset(seed);

  seed = static_cast<int> (list[n++]);
  random_unequal->reset(seed);

  next_reneighbor = (bigint) ubuf(list[n++]).i;

  ntranslation_attempts  = list[n++];
  ntranslation_successes = list[n++];
  nrotation_attempts     = list[n++];
  nrotation_successes    = list[n++];

  bigint ntimestep_restart = (bigint) ubuf(list[n++]).i;
  if (ntimestep_restart != update->ntimestep)
    error->all(FLERR,"Must not reset timestep when restarting fix vmmc");
}

/* ----------------------------------------------------------------------
   extract variable which stores whether MC is active or not
     active = MC moves are taking place
     not active = normal MD is taking place
   extract variable which stores index of exclusion group
------------------------------------------------------------------------- */

void *FixVMMC::extract(const char *name, int &dim)
{
  if (strcmp(name,"mc_active") == 0) {
    dim = 0;
    return (void *) &mc_active;
  }
  if (strcmp(name,"exclusion_group") == 0) {
    dim = 0;
    return (void *) &exclusion_group;
  }
  return nullptr;
}
