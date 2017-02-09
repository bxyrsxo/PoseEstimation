#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common_headers.h>
#include <pcl/pcl_macros.h>
#include <pcl/console/print.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/property_map/property_map.hpp>

using namespace std;
using namespace boost;

typedef adjacency_list < listS, vecS, undirectedS,no_property, property < edge_weight_t, int > > graph_t;
typedef graph_traits < graph_t >::vertex_descriptor vertex_descriptor;
typedef std::pair<int, int> Edge;

void createGraph( pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr in, std::vector<Edge>& vec_edge, std::vector<int>& vec_weight);
void Source_Point( pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr in, pcl::PointXYZ& pt, int& index);
int dijkstra_search( Edge* edge_array, int* weight, const int num_arcs, const int num_nodes, const int source_index);


boost::shared_ptr<pcl::visualization::PCLVisualizer>
rgbVis (pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr cloud)
{
	// -----Open 3D viewer and add point cloud-----
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb(cloud); 
	viewer->setBackgroundColor (0, 0, 0);
	viewer->addPointCloud<pcl::PointXYZRGBA> (cloud, rgb, "sample cloud");
	// viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
	// viewer->addCoordinateSystem (1.0);
	viewer->initCameraParameters ();
	
	return (viewer);
}
	
int
main(int argc, char** argv)
{
/*	
	// read point cloud, image and label files
	string pointcloud_filename = "./test1.pcd";
		
	// point cloud
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::io::loadPCDFile (pointcloud_filename, *cloud);

	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud1 (new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::io::loadPCDFile ("./p0.pcd", *cloud1);
	
	// point cloud visualization
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
    viewer = rgbVis( cloud);
	viewer->addPointCloud(cloud1);

	// find the source point
	pcl::PointXYZ source_pt, end_pt;
	int source_index;
	int end_pt_index;
	Source_Point( cloud, source_pt, source_index);
//	viewer->addSphere( source_pt, 0.05, 1, 1, 0);

	// create the map
	std::vector<Edge>   vec_edge;
	std::vector<int> vec_weight;
	createGraph( cloud, vec_edge, vec_weight); 
	
	Edge* edge_array;
	int* weight;
	int num_arcs = vec_edge.size();
	int num_vertex = cloud->size();

	edge_array = new Edge   [num_arcs];
	weight     = new int [num_arcs];

	for( int j = 0; j < num_arcs; j++)
	{
		edge_array[j] = vec_edge[j];
		weight[j] = vec_weight[j];
	}

	pcl::ScopeTime t ("dijkstra algorithm");
	for( int i = 0; i < 5; i++)
	{
		end_pt_index = dijkstra_search( edge_array, weight, num_arcs, num_vertex, source_index);
		end_pt.x = cloud->points[ end_pt_index].x;
		end_pt.y = cloud->points[ end_pt_index].y;
		end_pt.z = cloud->points[ end_pt_index].z;
		
		std::string name("end");
        std::stringstream ss;
		ss<<i;
		name = name + ss.str();

		viewer->addSphere( end_pt, 0.05, 1, 1, 0, name);
		source_index = end_pt_index;
	}
	
	int count = 0;
	for( int i = 0 ; i < num_arcs; i++)
		if( weight[i] == 0)
			count++;

	cout<<count<<endl;

    while (!viewer->wasStopped ())
    {
		viewer->spinOnce (100);
		boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }
*/	
	return 0;
}

void createGraph( pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr in, std::vector<Edge>& vec_edge, std::vector<int>& vec_weight)
{
	pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree;
    kdtree.setInputCloud ( in);
	pcl::PointXYZRGBA searchPoint;

	int i, j;
	int size = in->size();

	Edge tmp;
	for( j = 0; j < size; j++)
	{
		tmp.first = j;
		searchPoint.x = in->points[j].x; 
		searchPoint.y = in->points[j].y; 
		searchPoint.z = in->points[j].z; 
		
		int K = 6;
		std::vector<int> pointIdxNKNSearch(K);
		std::vector<float> pointNKNSquaredDistance(0.20);

		if ( kdtree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
		{
			for( i = 0; i < K; i++)
			{
				tmp.second = pointIdxNKNSearch[i];
				Eigen::Vector3d v1( in->points[ tmp.first].x,  in->points[ tmp.first].y,  in->points[ tmp.first].z);
				Eigen::Vector3d v2( in->points[ tmp.second].x, in->points[ tmp.second].y, in->points[ tmp.second].z);
				Eigen::Vector3d v3 = v1-v2;

				vec_edge.push_back( tmp);
				vec_weight.push_back( static_cast<int> (v3.norm()*1000));
			}
		}
	}

	
}

void Source_Point( pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr in, pcl::PointXYZ& pt, int& index)
{
	Eigen::Vector4f centroid;
	pcl::compute3DCentroid( *in, centroid);

	pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree;
    kdtree.setInputCloud ( in);
	pcl::PointXYZRGBA searchPoint;

	searchPoint.x = centroid(0); 
	searchPoint.y = centroid(1); 
	searchPoint.z = centroid(2); 

	std::vector<int> pointIdxNKNSearch(1);
    std::vector<float> pointNKNSquaredDistance(0.5);

	if ( kdtree.nearestKSearch (searchPoint, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
	{
		pt.x = in->points[ pointIdxNKNSearch[0] ].x;
		pt.y = in->points[ pointIdxNKNSearch[0] ].y;
		pt.z = in->points[ pointIdxNKNSearch[0] ].z;
		index = pointIdxNKNSearch[0];
	}
	else
		std::cout<<"K Nearest Neighbor Search Failed!"<<std::endl;	

	
}

int dijkstra_search( Edge* edge_array, int* weight, const int num_arcs, const int num_nodes, const int source_index)
{
	graph_t g(edge_array, edge_array + num_arcs, weight, num_nodes);
	property_map<graph_t, edge_weight_t>::type weightmap = get(edge_weight, g);
	std::vector<vertex_descriptor> p(num_vertices(g));
	std::vector<int> d(num_vertices(g));  


	vertex_descriptor s = vertex(source_index, g);
	dijkstra_shortest_paths(g, s,
	predecessor_map(boost::make_iterator_property_map(p.begin(), get(boost::vertex_index, g))).
	distance_map(boost::make_iterator_property_map(d.begin(), get(boost::vertex_index, g))));
	
	graph_traits < graph_t >::vertex_iterator vi, vend;

	int end_point = 0;
	int longest_path = 0;

	for (boost::tie(vi, vend) = vertices(g); vi != vend; ++vi) 
	{
		if( d[*vi] > longest_path && d[*vi] < 10000000 )
		{
			longest_path = d[*vi];
			end_point = *vi;			
		}
	}
	
	int	end_pt_index = end_point;
	cout<<"longest path: "<<longest_path<<", vectex:"<<end_point<<endl;
	
	while( end_point != source_index)
	{
		for( int q = 0; q < num_arcs; q++)
			if( edge_array[q].first == p[end_point] && edge_array[q].second == end_point)
			{
				weight[q] = 0;
				break;
			}
		
		end_point = p[end_point];
	}
	
	return end_pt_index;
}
