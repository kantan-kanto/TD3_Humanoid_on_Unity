using System;
using System.Collections.Generic;
using System.Net.Sockets;
using UnityEngine;
using UnityEditor;
using UnityEngine.Serialization;


public class Env : MonoBehaviour
{
    public GameObject HumanoidModel;
    public GameObject Target;
    public GameObject Floor;
    FloorCol script;
    
    const int num_objects = 17+1;
    const int num_joints = 11;
    const int dim_states = 8 + num_joints*2 + 2; // more8 + j22 + feet_contact2 = 32
    const int dim_actions = num_joints;

    GameObject[] game_objects = new GameObject[num_objects];
    Rigidbody[] rigid_bodies = new Rigidbody[num_objects];
    float[] game_objects_x = new float[num_objects];
    float[] game_objects_y = new float[num_objects];
    float[] game_objects_z = new float[num_objects];
    float[] game_objects_rx = new float[num_objects];
    float[] game_objects_ry = new float[num_objects];
    float[] game_objects_rz = new float[num_objects];
    float[] rigid_bodies_vx = new float[num_objects];
    float[] rigid_bodies_vy = new float[num_objects];
    float[] rigid_bodies_vz = new float[num_objects];
    float[] rigid_bodies_wx = new float[num_objects];
    float[] rigid_bodies_wy = new float[num_objects];
    float[] rigid_bodies_wz = new float[num_objects];
   

    float[] joint_position = new float[num_joints];    
    float[] joint_velocity = new float[num_joints];
    HingeJoint[] indexed_joints = new HingeJoint[num_joints];


    float reward;
    float done;
    float step;
    float countdown;

    float[] parts_data = new float[num_objects*12];
    float[] init_parts_data = new float[num_objects*12];
    float[] joints_data = new float[num_joints*2];
    float[] init_joints_data = new float[num_joints*2];
    float[] diff_joints_data = new float[num_joints*2];
    float[] data_Out = new float[dim_states+2]; //34
    float[] actions = new float[dim_actions];


    public bool stop_flag;
    public string ip = "127.0.0.1";
    public int port = 60000;
    private Socket client;
    [SerializeField]
    private float[] dataOut, dataIn;
    protected float[] ServerRequest(float[] dataOut)
    {
        this.dataOut = dataOut;
        this.dataIn = SendAndReceive(dataOut);
        return this.dataIn;
    }

    private float[] SendAndReceive(float[] dataOut)
    {
        //initialize socket
        float[] floatsReceived;
        client = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        client.Connect(ip, port);
        if (!client.Connected) {
            Debug.LogError("Connection Failed");
            return null;
        }

        //convert floats to bytes, send to port
        var byteArray = new byte[dataOut.Length * 4];
        Buffer.BlockCopy(dataOut, 0, byteArray, 0, byteArray.Length);
        client.Send(byteArray);

        //allocate and receive bytes
        byte[] bytes = new byte[4000];
        int idxUsedBytes = client.Receive(bytes);
        //print(idxUsedBytes + " new bytes received.");

        //convert bytes to floats
        floatsReceived = new float[idxUsedBytes/4];
        Buffer.BlockCopy(bytes, 0, floatsReceived, 0, idxUsedBytes);

        client.Close();
        return floatsReceived;
    }

    void Start () 
    {
        game_objects[0] = HumanoidModel.transform.Find( "torso1" ).gameObject;
        game_objects[1] = HumanoidModel.transform.Find( "head" ).gameObject;
        game_objects[2] = HumanoidModel.transform.Find( "uwaist" ).gameObject;
        game_objects[3] = HumanoidModel.transform.Find( "lwaist" ).gameObject;
        game_objects[4] = HumanoidModel.transform.Find( "butt" ).gameObject;
        game_objects[5] = HumanoidModel.transform.Find( "right_thigh1" ).gameObject;
        game_objects[6] = HumanoidModel.transform.Find( "right_shin1" ).gameObject;
        game_objects[7] = game_objects[6].transform.Find( "right_foot" ).gameObject;
        game_objects[8] = HumanoidModel.transform.Find( "left_thigh1" ).gameObject;
        game_objects[9] = HumanoidModel.transform.Find( "left_shin1" ).gameObject;
        game_objects[10] = game_objects[9].transform.Find( "left_foot" ).gameObject;
        game_objects[11] = HumanoidModel.transform.Find( "right_uarm1" ).gameObject;
        game_objects[12] = HumanoidModel.transform.Find( "right_larm" ).gameObject;
        game_objects[13] = HumanoidModel.transform.Find( "right_hand" ).gameObject;
        game_objects[14] = HumanoidModel.transform.Find( "left_uarm1" ).gameObject;
        game_objects[15] = HumanoidModel.transform.Find( "left_larm" ).gameObject;
        game_objects[16] = HumanoidModel.transform.Find( "left_hand" ).gameObject;
        game_objects[17] = Target;

        for (int i = 0; i < num_objects; i++) 
        {
            rigid_bodies[i] = game_objects[i].GetComponent<Rigidbody>();
        }

        HingeJoint[] Uwaist_Joint = game_objects[2].GetComponents<HingeJoint>();
        HingeJoint[] Lwaist_Joint = game_objects[3].GetComponents<HingeJoint>();
        HingeJoint[] Butt_Joint = game_objects[4].GetComponents<HingeJoint>();
        HingeJoint[] RightThigh1_Joint = game_objects[5].GetComponents<HingeJoint>();
        HingeJoint[] RightShin1_Joint = game_objects[6].GetComponents<HingeJoint>();
        HingeJoint[] LeftThigh1_Joint = game_objects[8].GetComponents<HingeJoint>();
        HingeJoint[] LeftShin1_Joint = game_objects[9].GetComponents<HingeJoint>();
        HingeJoint[] RightUarm1_Joint = game_objects[11].GetComponents<HingeJoint>();
        HingeJoint[] RightLarm_Joint = game_objects[12].GetComponents<HingeJoint>();
        HingeJoint[] LeftUarm1_Joint = game_objects[14].GetComponents<HingeJoint>();
        HingeJoint[] LeftLarm_Joint = game_objects[15].GetComponents<HingeJoint>();

        indexed_joints[0] = Uwaist_Joint[0];         // z -45 45       -45 45
        indexed_joints[1] = Lwaist_Joint[0];         // y -75 30?      -75 30
        indexed_joints[2] = Butt_Joint[0];           // x -35 35       -35 35

        indexed_joints[3] = RightThigh1_Joint[0];    //x- -25  5               
        // indexed_joints[4] = RightThigh1_Joint[1]; //y+ -35  60       
        // indexed_joints[5] = RightThigh1_Joint[2]; //z- -120 20      -120 20

        indexed_joints[4] = RightShin1_Joint[0];     //z+ -2  160  z-1 -160 2

        indexed_joints[5] = LeftThigh1_Joint[0];     //x+ -5   25        
        // indexed_joints[8] = LeftThigh1_Joint[1];  //y- -60  35
        // indexed_joints[9] = LeftThigh1_Joint[2];  //z- -120 20      -120 20

        indexed_joints[6] = LeftShin1_Joint[0];      //z+ -2  160  z-1 -160 2

        indexed_joints[7] = RightUarm1_Joint[0];     //x- -85  60
        // indexed_joints[12] = RightUarm1_Joint[1]; //z- -85  60      -85 60
        indexed_joints[8] = RightLarm_Joint[0];      //x  -90  50      -90 50

        indexed_joints[9] = LeftUarm1_Joint[0];      //x+ -60  85
        // indexed_joints[15] = LeftUarm1_Joint[1];  //z- -85  60      -85 60
        indexed_joints[10] = LeftLarm_Joint[0];      //x  -50  90  x-1 -90 50


        reward = 0;
        done = 0;
        step = 0;
        countdown = 1000;
        script = Floor.GetComponent<FloorCol>();


        StockPartsStates();
        StockJointsStates();  

        parts_data.CopyTo(init_parts_data, 0);
        joints_data.CopyTo(init_joints_data, 0);
    }

    void FixedUpdate()
    {
        StockPartsStates();
        StockJointsStates();
        StockOutputData();


        // actions = ServerRequest(data_Out);
        System.Random cRandom = new System.Random();
        for (int i = 0; i < num_joints; i++)
        {
            double dRandom = (cRandom.NextDouble() * 2.0) - 1.0;
            float fRandom = (float)dRandom;
            // if(i != 5 && i != 9){
            //     actions[i] = 0f; //fRandom; //-1f;
            // } else {
            //     actions[i] = -1f; //fRandom; //-1f;
            // }
            actions[i] = fRandom;
        }

        // HingeJoint General
        float[] min_limit = new float[num_joints]   {-45, -75, -35, -120, -160, -120, -160, -85, -90, -85, -90}; 
        float[] max_limit = new float[num_joints]   {45, 30, 35, 20, 2, 20, 2, 60, 50, 60, 50};
        for (int i = 0; i < num_joints; i++)
        {
            JointLimits[] Limits = new JointLimits[num_joints];
            Limits[i] = indexed_joints[i].limits;
            Limits[i].min = min_limit[i];
            Limits[i].max = max_limit[i];
            Limits[i].bounciness = 0f;
            Limits[i].bounceMinVelocity = 0f;
            indexed_joints[i].limits = Limits[i];
            indexed_joints[i].useLimits = true;
            indexed_joints[i].enableCollision = false;
            // Debug.Log(String.Format("idx{0}: R= {1:f}, V= {2:f}, Axis={3}", i, joints_data[i*2], joints_data[i*2+1], indexed_joints[i].axis));
        }


        // HingeJoint Motor
        JointMotor[] Motors = new JointMotor[num_joints];
        float[] motor_power = new float[num_joints] {100, 100, 100, 300, 200, 300, 200,  75,  75,  75,  75};
        for (int i = 0; i < num_joints; i++) //(int i = 5; i < 6; i++)//
        {
            Motors[i] = indexed_joints[i].motor;
            Motors[i].targetVelocity = Math.Sign(actions[i]) * motor_power[i] * 2f;
            Motors[i].force = 0.41f * Math.Abs(actions[i]) * 1000f;
            Motors[i].freeSpin = false;
            indexed_joints[i].motor = Motors[i];
            indexed_joints[i].useMotor = true;
            // Debug.Log(String.Format("idx{0}: R= {1:f}, V= {2:f}, Axis={3}", i, joints_data[i*2], joints_data[i*2+1], indexed_joints[i].axis));
        }

        // HingeJoint Spring
        // JointSpring[] Springs = new JointSpring[num_joints];
        // for (int i = 0; i < num_joints; i++) 
        // {
        //     Springs[i] = indexed_joints[i].spring;
        //     Springs[i].spring = 1000;
        //     Springs[i].damper = 10;
        //     Springs[i].targetPosition = init_joints_data[i*2];
        //     indexed_joints[i].spring = Springs[i];
        //     Debug.Log(String.Format("idx{0}: R= {1:f}, V= {2:f}, Axis={3}", i, joints_data[i*2], joints_data[i*2+1], indexed_joints[i].axis));
        // }


        if ((countdown <= 0) || (stop_flag)) //achieve the goal
        {
            done = 9;
            Quit();
        }
        else if (step > 1024 || game_objects_y[0] < 0.2f)
        {
            reward += 1;
            done = 1;
            Reset();
        }
        else
        {
            reward += 1;
            step += 1;
        } 
    }

    void Reset()
    {
        StockOutputData();
        // actions = ServerRequest(data_Out);
        SetInitPartsStates();
        countdown -= 1;
        reward = 0;
        done = 0;
        step = 0;
    }

    void Quit() 
    {
        StockOutputData();
        // ServerRequest(data_Out);

        #if UNITY_EDITOR
        EditorApplication.isPlaying = false;
        #elif UNITY_STANDALONE
        Application.Quit();
        #endif
    }


    void StockPartsStates()
    {
        for (int i = 0; i < num_objects; i++) 
        {
            game_objects_x[i] = game_objects[i].transform.position.x;
            game_objects_y[i] = game_objects[i].transform.position.y;
            game_objects_z[i] = game_objects[i].transform.position.z;
            game_objects_rx[i] = game_objects[i].transform.localEulerAngles.x;
            game_objects_ry[i] = game_objects[i].transform.localEulerAngles.y;
            game_objects_rz[i] = game_objects[i].transform.localEulerAngles.z;
            rigid_bodies_vx[i] = rigid_bodies[i].velocity.x;
            rigid_bodies_vy[i] = rigid_bodies[i].velocity.y;
            rigid_bodies_vz[i] = rigid_bodies[i].velocity.z;
            rigid_bodies_wx[i] = rigid_bodies[i].angularVelocity.x;
            rigid_bodies_wy[i] = rigid_bodies[i].angularVelocity.y;
            rigid_bodies_wz[i] = rigid_bodies[i].angularVelocity.z;
        }
        for (int i = 0; i < num_objects; i++) 
        {
            parts_data[i*12+0] = game_objects_x[i];
            parts_data[i*12+1] = game_objects_y[i];
            parts_data[i*12+2] = game_objects_z[i];
            parts_data[i*12+3] = game_objects_rx[i];
            parts_data[i*12+4] = game_objects_ry[i];
            parts_data[i*12+5] = game_objects_rz[i];
            parts_data[i*12+6] = rigid_bodies_vx[i];
            parts_data[i*12+7] = rigid_bodies_vy[i];
            parts_data[i*12+8] = rigid_bodies_vz[i];
            parts_data[i*12+9] = rigid_bodies_wx[i];
            parts_data[i*12+10] = rigid_bodies_wy[i];
            parts_data[i*12+11] = rigid_bodies_wz[i];
        }
    }

    void SetInitPartsStates()
    {
        for (int i = 0; i < num_objects; i++) 
        {
            game_objects[i].transform.position = new Vector3(init_parts_data[i*12+0], init_parts_data[i*12+1], init_parts_data[i*12+2]);
            game_objects[i].transform.localEulerAngles = new Vector3(init_parts_data[i*12+3], init_parts_data[i*12+4], init_parts_data[i*12+5]);
            rigid_bodies[i].velocity = new Vector3(init_parts_data[i*12+6], init_parts_data[i*12+7], init_parts_data[i*12+8]);
            rigid_bodies[i].angularVelocity = new Vector3(init_parts_data[i*12+9], init_parts_data[i*12+10], init_parts_data[i*12+11]);
        }        
        // Floor.transform.position = new Vector3(0f, -1.57f, 0f);
        // Floor.transform.eulerAngles = new Vector3(0f, 0f, 0f);
        // FloorRB.velocity = new Vector3(0f, 0f, 0f);
        // FloorRB.angularVelocity = new Vector3(0f, 0f, 0f);
    }

    float CurrentRelativePosition(HingeJoint hinge_joint)
    {
        float pos = hinge_joint.angle;
        if (pos < -180f) pos += 360f;   
        if (pos > 180f) pos -= 360f;
        return pos;
    }

    float CurrentRelativeVelosity(HingeJoint hinge_joint)
    {
        float vel = hinge_joint.velocity;
        return vel;
    }

    void StockJointsStates()
    {
        for (int i = 0; i < num_joints; i++) 
        {
            joint_position[i] = CurrentRelativePosition(indexed_joints[i]);
            joint_velocity[i] = CurrentRelativeVelosity(indexed_joints[i]);
        }

        for (int i = 0; i < num_joints; i++) 
        {
            joints_data[i*2] = joint_position[i];
            joints_data[i*2+1] = joint_velocity[i];
        }
    }

    void StockOutputData()
    {
        // more 8
        double angle_to_target = Math.Atan2(game_objects_z[17] - game_objects_z[0], game_objects_x[17] - game_objects_x[0]);
        double yaw = game_objects_ry[0] * Math.PI / 180f;
        Vector2 torso_v = new Vector2(rigid_bodies_vx[0], rigid_bodies_vz[0]);
        double torso_v_length = torso_v.magnitude;
        double torso_v_angle = Math.Atan2(rigid_bodies_vx[0] - game_objects_z[0], rigid_bodies_vx[0]);
        float sin_angle_to_target = (float)Math.Sin(angle_to_target);
        float cos_angle_to_target = (float)Math.Cos(angle_to_target);
        float vx = (float)(torso_v_length * Math.Cos(torso_v_angle - yaw));
        float vy = rigid_bodies_vy[0];
        float vz = (float)(torso_v_length * Math.Sin(torso_v_angle - yaw));
        data_Out[0] = game_objects_y[0] - init_parts_data[0*12+1];
        data_Out[1] = sin_angle_to_target;
        data_Out[2] = cos_angle_to_target;
        data_Out[3] = 0.3f * vx;
        data_Out[4] = 0.3f * vy;
        data_Out[5] = 0.3f * vz;
        data_Out[6] = game_objects_rx[0] * (float)Math.PI / 180f;
        data_Out[7] = game_objects_rz[0] * (float)Math.PI / 180f; 
        
        // j 11 * 2
        for(int i = 0; i < num_joints*2; i++)
        {
            data_Out[i+8] = joints_data[i];
            if(i % 2 == 0) data_Out[i] /= 180f;
            if(i % 2 == 1) data_Out[i] /= 600f; // Motors[i].targetVelocity = Math.Sign(actions[i]) * motor_power[i] * 2f;
        }

        // feet contact 2
        data_Out[dim_states-2] = script.feet_contact[0];
        data_Out[dim_states-1] = script.feet_contact[1];
        // Debug.Log(String.Format("Feet Contacts=[{0}, {1}], Ry={2:f}, Ly={3:f}, Step={4}", data_Out[dim_states-2], data_Out[dim_states-1], game_objects_y[7], game_objects_y[10], step));
        script.colList.Clear();

        // other
        data_Out[dim_states+0] = reward;
        data_Out[dim_states+1] = done;
    }
}