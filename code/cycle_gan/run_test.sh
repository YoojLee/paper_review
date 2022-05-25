set -ex # 에러 났을 경우 멈춤 -e / -x 쉘스크립트 추적 (명령 수행하기에 앞서서 해당 커맨드 출력해줌) -> 같이 사용하면 에러 디버깅하기가 매우 수월.
python code/cycle_gan/test.py --is_train False --data_root_A code/cycle_gan/horse2zebra/testA --data_root_B code/cycle_gan/horse2zebra/testB \
                              --last_checkpoint_dir code/cycle_gan/weights/no_reflect_padding --exp_name no_reflect_padding --load_epoch 200