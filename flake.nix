{
  inputs = {
    # We use an out of date nixpkgs, as we need an old symengine (<0.14) for pycalphad
    nixpkgs.url = github:NixOS/nixpkgs/nixos-25.11;
    oldnixpkgs.url = github:NixOS/nixpkgs/nixos-25.11;
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, oldnixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        venvDir = "./env";

        oldpkgs = import oldnixpkgs {
          inherit system;
        };

        overlay = final: prev: {
          symengine = oldpkgs.symengine;
        };

        pkgs = import nixpkgs {
          inherit system;
          overlays = [ overlay ];
        };

        coinhsltar = ./coinhsl-archive-2024.05.15.tar;
        
        simcem = (import (pkgs.fetchgit {
          url="https://github.com/simcem/SimCem.git"; 
          rev="31e76a542c28df8061f608b598ef7b5bb63de2a7";
          hash="sha256-N9R2hW6rkZo8rx24512Vog0wfTOplykOtl+xCyUw7Zk=";
          fetchSubmodules=true;
          # Because we need submodules, we need to workaround it not being deterministic
          # See https://github.com/NixOS/nixpkgs/issues/100498#issuecomment-1846499310
          leaveDotGit = true;
          postFetch = ''
            rm -rf $out/.git
            cp ${coinhsltar} ./coinhsl-archive-2024.05.15.tar
          '';
        }) { inherit pkgs; }).simcem;

        # Need to wait for pycalphad to update to symengine>=0.14!
        pycalphad_version = "0.11.1";
        pycalphad = pkgs.python3Packages.buildPythonPackage rec {
          format = "pyproject";
          name = "pycalphad";
          src = pkgs.fetchFromGitHub {
            owner = "pycalphad";
            repo = "pycalphad";
            rev = "${pycalphad_version}";
            sha256 = "sha256-A0bmTzqCSGuMwzfFvG4dxhVwJT6JUIyXm8suTBFsgO4=";
          };
          version = pycalphad_version;
          buildInputs = with pkgs.python3.pkgs; [
            python
            setuptools
            cython
            setuptools-scm
          ];
          propagatedBuildInputs = with pkgs.python3.pkgs; [
            numpy
            scipy
            symengine
            pandas
            pint
            matplotlib
            pyparsing
            pytest
            pytest-cov
            tinydb
            xarray
          ];
        };
      in
      {
        devShells.default = pkgs.mkShell {
          env = pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
            # Python libraries often load native shared objects using dlopen(3).
            # Setting LD_LIBRARY_PATH makes the dynamic library loader aware of libraries without using RPATH for lookup.
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath pkgs.pythonManylinuxPackages.manylinux1;
          };

          buildInputs = with pkgs; [
            (python3.withPackages (ps: with ps; [
              uv
              pip
              simcem
              pandas
              pint
              plotly
              streamlit
              scipy
              #pycalphad
              xlsxwriter
            ]))
          ];

          shellHook = ''
            uv venv .venv --system-site-packages --clear
            uv sync
            source .venv/bin/activate
          '';
        };
      })  
  ;
}
