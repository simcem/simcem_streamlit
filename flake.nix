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

  postShellHook = ''
    PYTHONPATH=\$PWD/\${venvDir}/\${pkgs.python3.sitePackages}/:\$PYTHONPATH
    # pip install -r requirements.txt
  '';
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            bashInteractive
            git
            screen
            (python3.withPackages (ps: with ps; [
              uv
              pip
              pybind11
              venvShellHook
            ]))
            ipopt
            eigen
            boost
            sundials
          ];
          postShellHook = postShellHook;
        };
      })  
  ;
}
