let
  host-nixpkgs = import <nixpkgs> { };
  pkgs-src = host-nixpkgs.fetchFromGitHub {
    owner = "NixOS";
    repo = "nixpkgs";
    rev = "bc90fe1fbb74070450774cfe1645a02846761557";
    sha256 = "11xyhmzm2x998qlh7p5q9arcmw6wsxqhkixlg5gwx5ir3gdijvpi";
  };
  pkgs = import pkgs-src { };
in
  with pkgs;
  stdenv.mkDerivation {
    name = "genstatmod";
    version = "1";
    buildInputs = [
      biber
      gtk3
      python
      pythonPackages.cvxopt
      pythonPackages.h5py
      pythonPackages.matplotlib
      pythonPackages.numpy
      pythonPackages.scikitlearn
      pythonPackages.scipy
      pythonPackages.statsmodels
      pythonPackages.pygobject3
      (texlive.combine {
        inherit (texlive) collection-fontsrecommended scheme-small latexmk geometry moreverb biblatex logreq xstring svg was preprint tikzmark import pgfopts texcount beamertheme-metropolis;
      })

  # ---
    ];
  }
