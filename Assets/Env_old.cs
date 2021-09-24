using System;
using System.Collections.Generic;
using System.Net.Sockets;
using UnityEngine;
using UnityEditor;
using UnityEngine.Serialization;
//


public class Env_old : MonoBehaviour
{
    public GameObject HumanoidModel;
    
    const int num_objects = 17;
    const int num_joints = 17;
    const int dim_states = 8 + num_joints*2 + 2;
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
    float[] data_Out = new float[dim_states+2];
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
        game_objects[7] = HumanoidModel.transform.Find( "right_foot" ).gameObject;
        game_objects[8] = HumanoidModel.transform.Find( "left_thigh1" ).gameObject;
        game_objects[9] = HumanoidModel.transform.Find( "left_shin1" ).gameObject;
        game_objects[10] = HumanoidModel.transform.Find( "left_foot" ).gameObject;
        game_objects[11] = HumanoidModel.transform.Find( "right_uarm1" ).gameObject;
        game_objects[12] = HumanoidModel.transform.Find( "right_larm" ).gameObject;
        game_objects[13] = HumanoidModel.transform.Find( "right_hand" ).gameObject;
        game_objects[14] = HumanoidModel.transform.Find( "left_uarm1" ).gameObject;
        game_objects[15] = HumanoidModel.transform.Find( "left_larm" ).gameObject;
        game_objects[16] = HumanoidModel.transform.Find( "left_hand" ).gameObject;

        for (int i = 0; i < num_objects; i++) 
        {
            rigid_bodies[i] = game_objects[i].GetComponent<Rigidbody>();
        }

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

        indexed_joints[0] = Lwaist_Joint[0];
        indexed_joints[1] = Lwaist_Joint[1];    
        indexed_joints[2] = Butt_Joint[0];

        indexed_joints[3] = RightThigh1_Joint[0];
        indexed_joints[4] = RightThigh1_Joint[1];
        indexed_joints[5] = RightThigh1_Joint[2];  

        indexed_joints[6] = RightShin1_Joint[0];

        indexed_joints[7] = LeftThigh1_Joint[0];
        indexed_joints[8] = LeftThigh1_Joint[1];
        indexed_joints[9] = LeftThigh1_Joint[2];

        indexed_joints[10] = LeftShin1_Joint[0];

        indexed_joints[11] = RightUarm1_Joint[0];
        indexed_joints[12] = RightUarm1_Joint[1];
        indexed_joints[13] = RightLarm_Joint[0];

        indexed_joints[14] = LeftUarm1_Joint[0];
        indexed_joints[15] = LeftUarm1_Joint[1];
        indexed_joints[16] = LeftLarm_Joint[0];

        reward = 0;
        done = 0;
        step = 0;
        countdown = 10;


        GetPartsStates();
        StockPartsStates();
        GetJointsStates();
        StockJointsStates();  

        parts_data.CopyTo(init_parts_data, 0);
        joints_data.CopyTo(init_joints_data, 0);
    }

    void FixedUpdate()
    {
        GetPartsStates();
        StockPartsStates();
        GetJointsStates();
        StockJointsStates();
        DiffJointsStates();

        StockOutputData();


        // actions = ServerRequest(data_Out);
        System.Random cRandom = new System.Random();
        for (int i = 0; i < num_joints; i++)
        {
            double dRandom = (cRandom.NextDouble() * 2.0) - 1.0;
            float fRandom = (float)dRandom;
            actions[i] = fRandom; //-1f;
        }


        // HingeJoint General
        for (int i = 0; i < num_joints; i++)
        {
            if(i != 5 && i != 9) indexed_joints[i].enableCollision = false;
            else indexed_joints[i].enableCollision = true;
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
        //     Debug.Log(String.Format("idx{0}: R= {1:f}, V= {2:f}", i, diff_joints_data[i*2], diff_joints_data[i*2+1]));
        // }


        // HingeJoint Motor
        JointMotor[] Motors = new JointMotor[num_joints];
        for (int i = 0; i < num_joints; i++) Motors[i] = indexed_joints[i].motor;

        float[] motor_power = new float[num_joints] {100, 100, 100, 100, 100, 300,  200,  100, 100, 300,  200,  75,  75,  75,  75,  75,  75};
        float[] max_limit = new float[num_joints]   {45,  30,  35,  5,   35,  20,   -2,   5,   35,  20,   -2,   60,  60,  50,  85,  85,  50};   
        float[] min_limit = new float[num_joints]   {-45, -75, -35, -25, -60, -120, -160, -25, -60, -120, -160, -85, -85, -90, -60, -60, -90};         

        for (int i = 0; i < num_joints; i++)
        {
            Motors[i].targetVelocity = Math.Sign(actions[i]) * 100f;
            if (diff_joints_data[i*2] > max_limit[i]) Motors[i].targetVelocity = -100f;
            else if (diff_joints_data[i*2] < min_limit[i]) Motors[i].targetVelocity = 100f;
            Motors[i].force = motor_power[i] * 0.41f * Math.Abs(actions[i]);
            Motors[i].freeSpin = true;
            indexed_joints[i].useMotor = true;
            indexed_joints[i].motor = Motors[i];
            Debug.Log(String.Format("idx{0}: R= {1:f}, V= {2:f}", i, diff_joints_data[i*2], diff_joints_data[i*2+1]));
        }


        if ((countdown <= 0) || (stop_flag)) //achieve the goal
        {
            done = 9;
            Quit();
        }
        else if (step > 1024)
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

    void GetPartsStates()
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
    }
    void StockPartsStates()
    {
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
            double pos = Vector3.Dot(hinge_joint.axis, hinge_joint.transform.localEulerAngles);
            if (pos < -180)
            {
                pos = pos + 360;
            }       
            if (pos > 180)
            {
                pos = pos - 360;
            }

            float pos_ = (float)pos;

    return pos_;
    }
    float CurrentRelativeVelosity(HingeJoint hinge_joint)
    {
            double vel = hinge_joint.velocity;
            float vel_ = (float)vel;

    return vel_;
    }
    void GetJointsStates()
    {
        for (int i = 0; i < num_joints; i++) 
        {
            joint_position[i] = CurrentRelativePosition(indexed_joints[i]);
            joint_velocity[i] = CurrentRelativeVelosity(indexed_joints[i]);
        }                                                                                                                                                    
    }
    void StockJointsStates()
    {
        for (int i = 0; i < num_joints; i++) 
        {
            joints_data[i*2] = joint_position[i];
            joints_data[i*2+1] = joint_velocity[i];
        }
    }
    void DiffJointsStates()
    {
        GetJointsStates();
        StockJointsStates();
        for(int i = 0; i < num_joints*2; i++)
        {
            diff_joints_data[i] = joints_data[i] - init_joints_data[i];


            // if (i % 2 == 0 && diff_joints_data[i] < -180)
            // {
            //     diff_joints_data[i] = diff_joints_data[i] + 360;
            // }       
            // else if (i % 2 == 0 && diff_joints_data[i] > 180)
            // {
            //     diff_joints_data[i] = diff_joints_data[i] - 360;
            // }

        }
    }
    void StockOutputData()
    {
        // more
        
        // j
        for(int i = 8; i < num_joints*2; i++)
        {
            data_Out[i] = joints_data[i];
            if(i% 2 == 0) data_Out[i] /= 180f;
        }

        // feet contact


        // other
        data_Out[dim_states+0] = reward;
        data_Out[dim_states+1] = done;
    }
}