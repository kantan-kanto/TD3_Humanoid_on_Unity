using System;
using System.Collections.Generic;
using System.Net.Sockets;
using UnityEngine;
using UnityEditor;
using UnityEngine.Serialization;


public class FloorCol : MonoBehaviour
{
    public int[] feet_contact = new int[2];
    // List<GameObject> colList = new List<GameObject> ();
    // string[] colList = new string[];
    public List<string> colList = new List<string>();
    
    
    void OnCollisionStay(Collision col)
    {       
        colList.Add (col.gameObject.name);
        feet_contact[0] = 0;
        feet_contact[1] = 0;
        for (int i = 0; i < colList.Count; i++)
        {
            // Debug.Log(colList[i]);
            // Debug.Log(String.Format("idx{0}: Object Name={1}", i, colList[i]));
            if(colList[i] == "right_foot") feet_contact[0] = 1;
            if(colList[i] == "left_foot") feet_contact[1] = 1;
        }
    }
}