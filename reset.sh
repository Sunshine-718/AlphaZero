#! usr/bin/dash

confirm() {
    while true; do
        printf "$1 [y/n]: "
        read -r yn
        case $yn in
            [Yy]* ) break;;
            [Nn]* ) echo "Operation cancelled."; exit;;
            * ) echo "Please answer y or n.";;
        esac
    done
}

confirm "Are you sure you want to delete the 'runs' and 'params' directories?"

rm -rd runs
rm -rd params
mkdir params
mkdir runs
