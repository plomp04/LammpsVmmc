/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(vmmc,FixVMMC);
// clang-format on
#else

#ifndef LMP_FIX_VMMC_H
#define LMP_FIX_VMMC_H

#include "fix.h"

namespace LAMMPS_NS {

class FixVMMC : public Fix {
 public:
  FixVMMC(class LAMMPS *, int, char **);
  ~FixVMMC() override;
  int setmask() override;
  void init() override;
  void pre_exchange() override;
  double compute_vector(int) override;
  double memory_usage() override;
  void write_restart(FILE *) override;
  void restart(char *) override;
  void *extract(const char *, int &) override;

 private:
  int molecule_group, molecule_group_bit;
  int molecule_group_inversebit;
  int exclusion_group, exclusion_group_bit;
  int nvmmc_type, nevery, seed;
  int ncycles, nexchanges, nmcmoves;
  double patomtrans, pmctot;
  int ngas;                // # of gas atoms on all procs
  int ngas_local;          // # of gas atoms on this proc
  int ngas_before;         // # of gas atoms on procs < this proc
  int exchmode;            // exchange ATOM or MOLECULE
  int movemode;            // move ATOM or MOLECULE
  class Region *region;    // vmmc region
  char *idregion;          // vmmc region id
  bool pressure_flag;      // true if user specified reservoir pressure
  bool charge_flag;        // true if user specified atomic charge
  bool full_flag;          // true if doing full system energy calculations

  int groupbitall;            // group bitmask for inserted atoms
  int ngroups;                // number of group-ids for inserted atoms
  char **groupstrings;        // list of group-ids for inserted atoms
  int ngrouptypes;            // number of type-based group-ids for inserted atoms
  char **grouptypestrings;    // list of type-based group-ids for inserted atoms
  int *grouptypebits;         // list of type-based group bitmasks
  int *grouptypes;            // list of type-based group types
  double ntranslation_attempts;
  double ntranslation_successes;
  double nrotation_attempts;
  double nrotation_successes;

  int mc_active;              // 1 during MC trials, otherwise 0

  int vmmc_nmax;
  int max_region_attempts;
  double gas_mass;
  double reservoir_temperature;
  double tfac_insert;
  double chemical_potential;
  double displace;
  double max_rotation_angle;
  double beta, zz, sigma, volume;
  double pressure, fugacity_coeff, charge;
  double xlo, xhi, ylo, yhi, zlo, zhi;
  double region_xlo, region_xhi, region_ylo, region_yhi, region_zlo, region_zhi;
  double region_volume;
  double energy_stored;    // full energy of old/current configuration
  double *sublo, *subhi;
  int *local_gas_list;
  double **cutsq;
  imageint imagezero;
  double overlap_cutoffsq;    // square distance cutoff for overlap
  int overlap_flag;
  int max_ngas;
  int min_ngas;

  double energy_intra;

  class Pair *pair;

  class RanPark *random_equal;
  class RanPark *random_unequal;

  class Atom *model_atom;

  int triclinic;    // 0 = orthog box, 1 = triclinic

  class Compute *c_pe;

  // private methods

  void options(int, char **);

  void attempt_atomic_translation();
  void attempt_atomic_translation_full();

  double energy(int, int, tagint, double *);
  double energy_full();

  int pick_random_gas_atom();
  tagint pick_random_gas_molecule();
  void toggle_intramolecular(int);
  void update_gas_atoms_list();

};

}    // namespace LAMMPS_NS

#endif
#endif
