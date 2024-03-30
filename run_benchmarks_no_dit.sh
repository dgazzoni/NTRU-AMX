#!/bin/bash

CPU=$(sysctl -n machdep.cpu.brand_string | sed -e 's/.*\(M[0-9]\).*/\1/')
COOLDOWN_BEGINNING=60
COOLDOWN_MIDDLE=30

# https://stackoverflow.com/a/226724/523079
# echo Requirements and recommendations for running this script:
# echo - Run it from the root of the repository using sudo, which is required to access the CPU cycle counter.
# echo - Run it using the caffeinate command to prevent the system going to sleep during the run: sudo caffeinate ./run_ct_experiments.sh
# echo - Try to remove as many sources of variability as possible. We recommend closing down all apps, including menubar ones, and turning off WiFi and Bluetooth. If it is a laptop, we recommend running it connected to AC power, and turning off low power mode.
# echo
# echo Would you like to continue?
# select yn in "Yes" "No"; do
#     case $yn in
#         Yes ) break;;
#         No ) exit;;
#     esac
# done


# https://unix.stackexchange.com/a/389406/493812
if [ "$(id -u)" -ne 0 ]; then
        echo 'This script must be run by root'
        exit 1
fi

if [ -d speed_results_${CPU}_no_dit ]
then
    # https://stackoverflow.com/a/226724/523079
    echo "Existing results will be deleted. Are you sure?"
    select yn in "Yes" "No"; do
        case $yn in
            Yes ) rm -rf speed_results_${CPU}_no_dit; break;;
            No ) exit;;
        esac
    done
fi

mkdir speed_results_${CPU}_no_dit

rm -rf build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DUSE_FEAT_DIT=OFF -G Ninja ..
ninja

echo Waiting for the system to cool down for ${COOLDOWN_BEGINNING} seconds...
sleep ${COOLDOWN_BEGINNING}

for PARAMETER_SET in hps2048509 hps2048677 hps4096821 hrss701
do
    for ALLOC in stack mmap
    do
        if [ "$PARAMETER_SET" == "hps2048677" ] || [ "$PARAMETER_SET" == "hrss701" ]
        then
            IMPLS="NG21 CCHY23"
        else
            IMPLS="NG21"
        fi

        for IMPL in $IMPLS
        do
            if [ "$IMPL" == "NG21" ]
            then
                VARIANTS="amx neon"
            else
                if [ "$PARAMETER_SET" == "hps2048677" ]
                then
                    VARIANTS="amx tc tmvp"
                else
                    VARIANTS="amx tmvp"
                fi
            fi

            for VARIANT in $VARIANTS
            do
                SPEED_EXEC=speed_ntru${PARAMETER_SET}_${ALLOC}_${IMPL}_${VARIANT}
                echo "Benchmarking ${SPEED_EXEC}"
                
                # https://stackoverflow.com/questions/77711672/performance-of-cpu-only-code-varies-with-executable-file-name
                cp ${SPEED_EXEC} speed

                # Run twice to warm up; e.g. macOS needs to verify the code signature and is slower on the first run
                ./speed > /dev/null
                ./speed > /dev/null
                # Actual run
                ./speed > "../speed_results_${CPU}_no_dit/ntru:${PARAMETER_SET}:${ALLOC}:${IMPL}:${VARIANT}.txt"

                echo Waiting for the system to cool down for ${COOLDOWN_MIDDLE} seconds...
                sleep ${COOLDOWN_MIDDLE}

                SPEED_POLYMUL_EXEC=speed_polymul_ntru${PARAMETER_SET}_${ALLOC}_${IMPL}_${VARIANT}
                if [ -f ${SPEED_POLYMUL_EXEC} ]
                then
                    echo "Benchmarking ${SPEED_POLYMUL_EXEC}"

                    # https://stackoverflow.com/questions/77711672/performance-of-cpu-only-code-varies-with-executable-file-name
                    cp ${SPEED_POLYMUL_EXEC} speed

                    # Run twice to warm up; e.g. macOS needs to verify the code signature and is slower on the first run
                    ./speed > /dev/null
                    ./speed > /dev/null
                    # Actual run
                    ./speed \
                        > "../speed_results_${CPU}_no_dit/polymul:${PARAMETER_SET}:${ALLOC}:${IMPL}:${VARIANT}.txt"

                    echo Waiting for the system to cool down for ${COOLDOWN_MIDDLE} seconds...
                    sleep ${COOLDOWN_MIDDLE}
                fi
            done
        done
    done
done

for MISC_TEST in polymul_ntru polymul_scaling polymodmul_ntru polymodmul_scaling polymodmul_karatsuba_scaling blas
do
    SPEED_EXEC=speed_${MISC_TEST}
    echo "Benchmarking ${SPEED_EXEC}"
    
    # https://stackoverflow.com/questions/77711672/performance-of-cpu-only-code-varies-with-executable-file-name
    cp ${SPEED_EXEC} speed

    # Run twice to warm up; e.g. macOS needs to verify the code signature and is slower on the first run
    ./speed > /dev/null

    if [ "$MISC_TEST" == "blas" ]
    then
        # speed_blas runs for a longer time than the others, so insert a cooling-down period in between runs as well
        echo Waiting for the system to cool down for ${COOLDOWN_MIDDLE} seconds...
        sleep ${COOLDOWN_MIDDLE}
    fi

    ./speed > /dev/null

    if [ "$MISC_TEST" == "blas" ]
    then
        # speed_blas runs for a longer time than the others, so insert a cooling-down period in between runs as well
        echo Waiting for the system to cool down for ${COOLDOWN_MIDDLE} seconds...
        sleep ${COOLDOWN_MIDDLE}
    fi

    # Actual run
    ./speed > "../speed_results_${CPU}_no_dit/${MISC_TEST}.txt"

    echo Waiting for the system to cool down for ${COOLDOWN_MIDDLE} seconds...
    sleep ${COOLDOWN_MIDDLE}
done

cd ..

chown -R $(logname) build/ speed_results_${CPU}_no_dit/
