{ pkgs ? import <nixpkgs> {} }:

let python_with_deps = pkgs.python35.withPackages
      (ps: [ps.scipy ps.scikitlearn ps.matplotlib ps.pandas ps.seaborn ps.jupyter
            ps.pyqt4 # Needed only for matplotlib backend
            ps.xgboost
            ]);
in pkgs.stdenv.mkDerivation rec {
  name = "kaggle-sandbox";

  buildInputs = with pkgs; [ python_with_deps ];
}
