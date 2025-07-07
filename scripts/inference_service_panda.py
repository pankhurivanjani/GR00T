
import argparse
import numpy as np

from gr00t.eval.robot import RobotInferenceClient, RobotInferenceServer
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model checkpoint directory.",
        default="nvidia/GR00T-N1-2B",
    )
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        help="The embodiment tag for the model.",
        default="gr1",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        help="The name of the data config to use.",
        choices=list(DATA_CONFIG_MAP.keys()),
        default="gr1_arms_waist",
    )

    parser.add_argument("--port", type=int, help="Port number for the server.", default=5555)
    parser.add_argument(
        "--host", type=str, help="Host address for the server.", default="localhost"
    )
    # server mode
    parser.add_argument("--server", action="store_true", help="Run the server.")
    # client mode
    parser.add_argument("--client", action="store_true", help="Run the client")
    parser.add_argument("--denoising_steps", type=int, help="Number of denoising steps.", default=4)
    args = parser.parse_args()

    if args.server:
        # Create a policy
        # The `Gr00tPolicy` class is being used to create a policy object that encapsulates
        # the model path, transform name, embodiment tag, and denoising steps for the robot
        # inference system. This policy object is then utilized in the server mode to start
        # the Robot Inference Server for making predictions based on the specified model and
        # configuration.

        # we will use an existing data config to create the modality config and transform
        # if a new data config is specified, this expect user to
        # construct your own modality config and transform
        # see gr00t/utils/data.py for more details
        data_config = DATA_CONFIG_MAP[args.data_config]
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        policy = Gr00tPolicy(
            model_path=args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=args.embodiment_tag,
            denoising_steps=args.denoising_steps,
        )

        # Start the server
        server = RobotInferenceServer(policy, port=args.port)
        server.run()
        print(f"😀 server starts successfully, waiting for a request ... \n host: {args.host}, port: {args.port}")

    elif args.client:
        import time
        # TODO import real robot related API
        from real_robot.real_robot_env.robot.hardware_franka import FrankaArm, ControlType
        from real_robot.real_robot_env.robot.hardware_frankahand import FrankaHand
        from real_robot.real_robot_env.robot.hardware_cameras import AsynchronousDevice
        from real_robot.real_robot_env.robot.hardware_depthai import DepthAI
        from real_robot.utils.keyboard_input import NonBlockingKeyPress


        # This is useful for testing the server and client connection
        # Create a policy wrapper
        policy_client = RobotInferenceClient(host=args.host, port=args.port)
        print(f"host: {args.host}, port: {args.port}")

        print("Available modality config available:")
        modality_configs = policy_client.get_modality_config()
        print(f"modality_configs: {modality_configs.keys()}")

        # Making prediction...
        # - obs: video.ego_view: (1, 256, 256, 3)
        # - obs: state.gripper_state: (1, )
        # - obs: state.joint_pos: (1, 7)
        # - obs: state.joint_vel: (1, 7)

        # - action: action.gripper_state: (16, 1)
        # - action: action.joint_pos: (16, 7)
        # - action: action.joint_vel: (16, 7)
 
        # -------------------------
        # robot and camera creation
        # -------------------------
        p4 = FrankaArm(name='p4', ip_address='10.10.10.11', port=50053, control_type=ControlType.HYBRID_JOINT_IMPEDANCE_CONTROL)
        assert p4.connect(), f"Connection to {p4.name} failed"

        p4_hand = FrankaHand(name="p4_hand", ip_address='10.10.10.11', port=50054)
        assert p4_hand.connect(), f"Connection to {p4_hand.name} failed"

        camera = DepthAI(device_id="19443010B11CEF1200",height=256, width=256)
        assert camera.connect(), f"Connection to {camera.name} failed"

        delta_t = 0.05 
        max_width = 0.085

        print(" 🔍 Press 'n' to start inference or 'q' to quit inference")
        with NonBlockingKeyPress() as kp:
            quit = False
            while not quit:
                key = kp.get_data()
                if key == "q":
                    quit = True

                if key == "n":
                    print("🔄 Resetting robot...")
                    p4.reset()
                    p4_hand.reset()
                    #start the inference loop or stop

                    inference = True
                    while inference:
                        # Update input
                        key = kp.get_data()
                        if key == "s": #stop
                            print("🛑 Inference stopped.")
                            inference = False
                            break

                        # get the state
                        image = camera.get_sensors()["rgb"]   # BGR
                        gripper_width = p4_hand.get_sensors().item()
                        gripper_state = -1 if gripper_width < max_width/1.3 else 1
                        joint_state = p4.get_state()
                        joint_pos = joint_state.joint_pos
                        joint_vel = joint_state.joint_vel

                        obs = {
                            "video.ego_view": image,
                            "state.gripper_state": gripper_state,
                            "state.joint_pos": joint_pos,
                            "state.joint_vel": joint_vel,
                            "annotation.human.action.task_description": ["pick up the apple from the desktop and place it in the corner"],
                        }
                        print("🟡 Sending observation to server...")

                        predicted_action = policy_client.get_action(obs)
                        print("🟢 Received action from server.")

                        gripper_command = predicted_action["action.gripper_state"]
                        joint_pos_command = predicted_action["action.joint_pos"]
                        joint_vel_command = predicted_action["action.joint_vel"]

                        for i in range(joint_pos_command.shape[0]): # 16
                            p4.apply_commands(q_desired=joint_pos_command[i], qd_desired=joint_vel_command[i])
                            p4_hand.apply_commands(width=gripper_command[i])
                            time.sleep(delta_t) 
        
        print("⌛ Ending inference...")
        p4.close()
        p4_hand.close()
        camera.close()





        

