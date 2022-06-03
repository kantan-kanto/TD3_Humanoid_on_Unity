using System;
using System.Collections.Generic;
using System.Net.Sockets;
using UnityEngine;
using UnityEditor;
using UnityEngine.Serialization;


public class FootCol : MonoBehaviour
{
    public int[] contact = new int[2];
    // List<GameObject> colList = new List<GameObject> ();
    // string[] colList = new string[];
    public List<string> colList = new List<string>();
    
    
    void OnCollisionStay(Collision col)
    {       
        colList.Add (col.gameObject.name);
        contact[0] = 0;
        contact[1] = 0;
        for (int i = 0; i < colList.Count; i++)
        {
            // Debug.Log(colList[i]);
            if(colList[i] != "Floor") contact[0] = 1;
            // if(colList[i] == "left_shin1") contact[1] = 1;
            // Debug.Log(String.Format("idx{0}: Object Name={1} Penalty={2}", i, colList[i], contact[0]));
        }
    }
}