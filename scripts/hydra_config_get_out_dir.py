import hydra
import shutil
import os.path as osp

@hydra.main(config_path="../configs", config_name="base", version_base="1.2")
def main(FLAGS):
    print(FLAGS.out_dir)
    if osp.exists(FLAGS.out_dir):
        shutil.rmtree(FLAGS.out_dir)
    else:
        raise ValueError(f"{FLAGS.out_dir} does not exist, check `hydra.run.dir`")


if __name__ == '__main__':
    main()
