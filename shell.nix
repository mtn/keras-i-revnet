with import <nixpkgs> {};

let
  py = pkgs.python37;
in
stdenv.mkDerivation rec {
  name = "python-environment";

  buildInputs = [
    py
    py.pkgs.black
    py.pkgs.h5py
    py.pkgs.scikitlearn
    py.pkgs.matplotlib
    py.pkgs.tkinter
    py.pkgs.numpy
    py.pkgs.tensorflowWithCuda
  ];
}
