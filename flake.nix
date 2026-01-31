{
  inputs = {
    nixpkgs.url = github:NixOS/nixpkgs/nixos-25.11;
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }: 
    flake-utils.lib.eachDefaultSystem (system:
      let
        venvDir = "./env";

        pkgs = import nixpkgs {
          inherit system;
          #overlays = [ self.overlays.default ];
        };

        simcem = (import (pkgs.fetchgit {
          url="https://github.com/simcem/SimCem.git"; 
          rev="1c0371748c0fbec816e29f18fe91d6bc2fb60840";
          hash="sha256-LS+1RjkXMHTK+XhdAxkuPd/aeFeWCU+UuipDAcW+5rs=";
          fetchSubmodules=true;
          # Because we need submodules, we need to workaround it not being deterministic
          # See https://github.com/NixOS/nixpkgs/issues/100498#issuecomment-1846499310
          leaveDotGit = true;
          postFetch = ''
            rm -rf $out/.git
          '';
        }) { inherit pkgs; });
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            (python3.withPackages (ps: with ps; [
              uv
              pip
              simcem.simcem
              pandas
              pint
              plotly
              streamlit
              scipy
            ]))
          ];
        };
      })  
  ;
}
