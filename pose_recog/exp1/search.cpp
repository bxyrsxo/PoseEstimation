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
#include <pcl/segmentation/extract_clusters.h>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/property_map/property_map.hpp>

using namespace std;
using namespace boost;

typedef adjacency_list < listS, vecS, undirectedS,no_property, property < edge_weight_t, int > > graph_t;
typedef graph_traits < graph_t >::vertex_descriptor vertex_descriptor;
typedef std::pair<int, int> Edge;

class HumanPose
{
public:
	HumanPose()
	{
		lowerArmLength = 0.35;  
		upperArmLength = 0.35;
		lowerFeetLength = 0.48;
		upperFeetLength = 0.40;
		
		lowerArmRadius = 0.01;		
		upperArmRadius = 0.01;
		lowerFeetRadius = 0.01;
		upperFeetRadius = 0.01;
	}
		
	pcl::PointXYZ rHand, lHand;
	pcl::PointXYZ rElbow, lElbow;  
	pcl::PointXYZ rShoulder, lShoulder;
	pcl::PointXYZ rAnkle, lAnkle;
	pcl::PointXYZ rKnee, lKnee;
	pcl::PointXYZ rHip, lHip;
	
	double lowerArmLength, upperArmLength;
	double lowerFeetLength, upperFeetLength;
	double lowerArmRadius,  upperArmRadius;
	double lowerFeetRadius, upperFeetRadius;
};


void   createGraph( pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr in, std::vector<Edge>& vec_edge, std::vector<int>& vec_weight);
void   Source_Point( pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr in, pcl::PointXYZ& pt, int& index);
int    dijkstra_search( Edge* edge_array, int* weight, const int num_arcs, const int num_nodes, const int source_index);
void   seg_lpart ( pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& in, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& seg, pcl::PointXYZ& pt, double search_radius);
void   seg_upart ( pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& in, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& seg, pcl::PointXYZ& pt, pcl::PointXYZ& pt1, double search_radius);
int    get_min_sequence(Eigen::EigenSolver<Eigen::MatrixXd>& es);
void   Direction_Decision( Eigen::VectorXd& th, double& u, double& v, double& w, pcl::PointXYZ& J0, pcl::PointXYZ& Pi);
double angle_bt_vectors( pcl::PointXYZ& p1, pcl::PointXYZ& p2, pcl::PointXYZ& p3);
void   pose_recognition( pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& in, pcl::PointXYZ& pt, Eigen::Vector3d& vecA);
void   indirect_scheme ( Eigen::VectorXd& th, double& u, double& v, double& w);
void   indirect_scheme1( Eigen::VectorXd& th, double& u, double& v, double& w);
void   Direction_Decision ( double& u, double& v, double& w, pcl::PointXYZ& J0, pcl::PointXYZ& Pi);
void   Direction_Decision1( Eigen::VectorXd& th, double& u, double& v, double& w, pcl::PointXYZ& J0, pcl::PointXYZ& Pi);
void   show_visualizer( pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud, HumanPose& pose);
void   show_visualizer_points( pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud, HumanPose& pose);

boost::shared_ptr<pcl::visualization::PCLVisualizer>
rgbVis (pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr cloud)
{
	// -----Open 3D viewer and add point cloud-----
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb(cloud); 
	viewer->setBackgroundColor (1, 1, 1 );
	viewer->addPointCloud<pcl::PointXYZRGBA> (cloud, rgb, "sample cloud");
//	viewer->addCoordinateSystem (1.0);
//	viewer->initCameraParameters ();
	while (!viewer->wasStopped ())
    {
		viewer->spinOnce (100);
		boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }
	
	return (viewer);
}
	
int
main(int argc, char** argv)
{
	
	// read point cloud, image and label files
	string pointcloud_filename = "./voxel.pcd";
		
	// point cloud
	pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_cloud (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPCDFile(pointcloud_filename, *tmp_cloud);


	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
//	pcl::io::loadPCDFile (pointcloud_filename, *cloud);
	for( int i = 0; i < tmp_cloud->size(); i++)
	{
		pcl::PointXYZRGBA pt;
		pt.x = tmp_cloud->points[i].x;
		pt.y = tmp_cloud->points[i].y;
		pt.z = tmp_cloud->points[i].z;
		pt.r = 0.5;
		pt.g = 0.5;
		pt.b = 0.5;
	
		cloud->push_back( pt);
	}

	cout<<cloud->size()<<endl;

	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud1 (new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::io::loadPCDFile ("./human.pcd", *cloud1);
	
	// point cloud visualization
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
    viewer = rgbVis( cloud);
 
//	viewer->addPointCloud(cloud1);

	// find the source point
	pcl::PointXYZ source_pt, end_pt;
	int source_index;
	int end_pt_index;
	Source_Point( cloud, source_pt, source_index);
	viewer->addSphere( source_pt, 0.05, 1, 1, 0);

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
	
	std::vector<pcl::PointXYZ> vec_end_pt;
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
		vec_end_pt.push_back(end_pt);
	
	}


	// lack a fraction of algorithm to determine the feature point if it is head, hand or foot
	for( int i = 0; i < vec_end_pt.size(); i++)
		cout<<vec_end_pt[i].x<<", "<<vec_end_pt[i].y<<", "<<vec_end_pt[i].z<<endl;

	HumanPose pose;
	pose.rHand  = vec_end_pt[0];
	pose.lHand  = vec_end_pt[2];
	pose.rAnkle = vec_end_pt[3];
	pose.lAnkle = vec_end_pt[1];



	Eigen::Vector3d vecA;
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr seg ( new pcl::PointCloud<pcl::PointXYZRGBA>);
	// right lower arm
	seg_lpart( cloud1, seg, pose.rHand, pose.lowerArmLength);
	pose_recognition( seg, pose.rHand, vecA);
	pose.rElbow.x = vecA(0)*pose.lowerArmLength + pose.rHand.x;
	pose.rElbow.y = vecA(1)*pose.lowerArmLength + pose.rHand.y;
	pose.rElbow.z = vecA(2)*pose.lowerArmLength + pose.rHand.z;
	// left lower arm
//	seg_lpart( cloud, seg, pose.lHand, pose.lowerArmLength);
//	pose_recognition( seg, pose.lHand, vecA);
//	pose.lElbow.x = vecA(0)*pose.lowerArmLength + pose.lHand.x;
//	pose.lElbow.y = vecA(1)*pose.lowerArmLength + pose.lHand.y;
//	pose.lElbow.z = vecA(2)*pose.lowerArmLength + pose.lHand.z;

	// manual designation here, becase the point cloud becomes two part after segmenting from the background
	pose.lElbow.x = pose.rElbow.x;
	pose.lElbow.y = pose.rElbow.y;
	pose.lElbow.z = pose.rElbow.z;
	
	// right lower leg
	seg_lpart( cloud, seg, pose.rAnkle, pose.lowerFeetLength);
	pose_recognition( seg, pose.rAnkle, vecA);
	pose.rKnee.x = vecA(0)*pose.lowerFeetLength + pose.rAnkle.x;
	pose.rKnee.y = vecA(1)*pose.lowerFeetLength + pose.rAnkle.y;
	pose.rKnee.z = vecA(2)*pose.lowerFeetLength + pose.rAnkle.z;
	// left lower leg
	seg_lpart( cloud, seg, pose.lAnkle, pose.lowerFeetLength);
	pose_recognition( seg, pose.lAnkle, vecA);
	pose.lKnee.x = vecA(0)*pose.lowerFeetLength + pose.lAnkle.x;
	pose.lKnee.y = vecA(1)*pose.lowerFeetLength + pose.lAnkle.y;
	pose.lKnee.z = vecA(2)*pose.lowerFeetLength + pose.lAnkle.z;

	// right upper arm
	seg_upart( cloud1, seg, pose.rElbow, pose.rHand, pose.upperArmLength);
	pose_recognition( seg, pose.rElbow, vecA);
	pose.rShoulder.x = vecA(0)*pose.upperArmLength + pose.rElbow.x;
	pose.rShoulder.y = vecA(1)*pose.upperArmLength + pose.rElbow.y;
	pose.rShoulder.z = vecA(2)*pose.upperArmLength + pose.rElbow.z;

	// left upper arm
	seg_upart( cloud1, seg, pose.lElbow, pose.lHand, pose.upperArmLength);
	pose_recognition( seg, pose.rElbow, vecA);
	pose.lShoulder.x = vecA(0)*pose.upperArmLength + pose.lElbow.x;
	pose.lShoulder.y = vecA(1)*pose.upperArmLength + pose.lElbow.y;
	pose.lShoulder.z = vecA(2)*pose.upperArmLength + pose.lElbow.z;

	// to be fixed
	pose.lShoulder.x = pose.rShoulder.x;
	pose.lShoulder.y = pose.rShoulder.y;
	pose.lShoulder.z = pose.rShoulder.z;

	// right upper leg 
	seg_upart( cloud1, seg, pose.rKnee, pose.rAnkle, pose.upperFeetLength);
	pose_recognition( seg, pose.rKnee, vecA);
	pose.rHip.x = vecA(0)*pose.upperFeetLength + pose.rKnee.x;
	pose.rHip.y = vecA(1)*pose.upperFeetLength + pose.rKnee.y;
	pose.rHip.z = vecA(2)*pose.upperFeetLength + pose.rKnee.z;

	// left upper leg 
	seg_upart( cloud1, seg, pose.lKnee, pose.lAnkle, pose.upperFeetLength);
	pose_recognition( seg, pose.lKnee, vecA);
	pose.lHip.x = vecA(0)*pose.upperFeetLength + pose.lKnee.x;
	pose.lHip.y = vecA(1)*pose.upperFeetLength + pose.lKnee.y;
	pose.lHip.z = vecA(2)*pose.upperFeetLength + pose.lKnee.z;

	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr scene (new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::io::loadPCDFile ("./scene.pcd", *scene);
	show_visualizer_points(scene, pose);

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

void seg_lpart ( pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& in, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& seg, pcl::PointXYZ& end, double radius)
{
	seg->clear();
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr search ( new pcl::PointCloud<pcl::PointXYZRGBA>);
	 
	double offset = 0.08;
	double rsqu = (radius-offset)*(radius-offset);
	for( int i = 0; i < in->size(); i++)
	{
		pcl::PointXYZRGBA pt = in->points[i];
		if( (pt.x-end.x)*(pt.x-end.x)+(pt.y-end.y)*(pt.y-end.y)+(pt.z-end.z)*(pt.z-end.z) < rsqu)
			search->push_back(pt);
	}

    // Creating the KdTree object for the search method of the extraction
	pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGBA>);
	tree->setInputCloud (search);
	 
	std::vector<pcl::PointIndices> indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZRGBA> ec;
	ec.setClusterTolerance (0.06); // 2cm
	ec.setMinClusterSize (20);
	ec.setMaxClusterSize (100000);
	ec.setSearchMethod (tree);
	ec.setInputCloud (search);
	ec.extract (indices);
	 

	if( indices.size() == 1)
	{
		std::vector<int> vec = indices[0].indices;
		for( int i = 0; i < vec.size(); i++)
			seg->push_back( search->points[ vec[i]]);
	}
	else if( indices.size() == 2)
	{
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr c1 ( new pcl::PointCloud<pcl::PointXYZRGBA>);		
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr c2 ( new pcl::PointCloud<pcl::PointXYZRGBA>);	

		std::vector<int> vec = indices[0].indices;	
		std::vector<int> vec1 = indices[1].indices;

		pcl::PointXYZRGBA pt;
		pt.x = end.x;  pt.y = end.y;  pt.z = end.z;

		for( int i = 0; i < vec.size(); i++)
			c1->push_back( search->points[ vec[i]]);
		c1->push_back( pt);

		for( int i = 0; i < vec1.size(); i++)
			c2->push_back( search->points[ vec1[i]]);
		c2->push_back( pt);

		// calculate average distances of c1 and c2
		pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree, kdtree1;
		kdtree.setInputCloud ( c1);
		kdtree1.setInputCloud ( c2);
	
		int K = 10;
		std::vector<int> pointIdxNKNSearch(K);
		std::vector<int> pointIdxNKNSearch1(K);
		std::vector<float> pointNKNSquaredDistance(K);
		std::vector<float> pointNKNSquaredDistance1(K);
	
		float avg_c1 = 0, avg_c2 = 0;		
		if ( kdtree.nearestKSearch (pt, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
			for( int i = 0; i < K; i++)
				avg_c1 += pointNKNSquaredDistance[i];

		if ( kdtree1.nearestKSearch (pt, K, pointIdxNKNSearch1, pointNKNSquaredDistance1) > 0 )
			for( int i = 0; i < K; i++)
				avg_c2 += pointNKNSquaredDistance1[i];

		if( avg_c1 >= avg_c2)
			seg = c2;
		else
			seg = c1;
	}
	else
	{
		// to do
		cout<<"more than two clusters!"<<endl;
	}
}

void seg_upart ( pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& in, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& seg, pcl::PointXYZ& end, pcl::PointXYZ& end1, double radius)
{
	seg->clear();
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr search ( new pcl::PointCloud<pcl::PointXYZRGBA>);
	 
	double offset = 0.06;
	double rsqu  = (radius-offset)*(radius-offset);
	double rsqu1 = 0.01;
	for( int i = 0; i < in->size(); i++)
	{
		pcl::PointXYZRGBA pt = in->points[i];
		double d = (pt.x-end.x)*(pt.x-end.x)+(pt.y-end.y)*(pt.y-end.y)+(pt.z-end.z)*(pt.z-end.z);
		if( d < rsqu && d > rsqu1)
			search->push_back(pt);
	}

    	// Creating the KdTree object for the search method of the extraction
	pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGBA>);
	tree->setInputCloud (search);
	 
	std::vector<pcl::PointIndices> indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZRGBA> ec;
	ec.setClusterTolerance (0.06); // 2cm
	ec.setMinClusterSize (20);
	ec.setMaxClusterSize (100000);
	ec.setSearchMethod (tree);
	ec.setInputCloud (search);
	ec.extract (indices);

	if( indices.size() > 2)
	{


	}
	else if( indices.size() == 2)
	{	
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr c1 ( new pcl::PointCloud<pcl::PointXYZRGBA>);		
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr c2 ( new pcl::PointCloud<pcl::PointXYZRGBA>);	

		std::vector<int> vec = indices[0].indices;	
		std::vector<int> vec1 = indices[1].indices;
		
		double deviation = 0.05;
		for( int i = 0; i < vec.size(); i++)
			if( search->points[ vec[i]].y < end.y + deviation && search->points[ vec[i]].y > end.y - deviation)
				c1->push_back( search->points[ vec[i]]);

		for( int i = 0; i < vec1.size(); i++)
			if( search->points[ vec1[i]].y < end.y + deviation && search->points[ vec1[i]].y > end.y - deviation)
				c2->push_back( search->points[ vec1[i]]);

		// calculate centroid of c1 and c2
		pcl::PointXYZ vecQ, vecW, vecR;
		Eigen::Vector4f centroid1, centroid2;
		pcl::compute3DCentroid( *c1, centroid1);
		pcl::compute3DCentroid( *c2, centroid2);
			
		pcl::PointXYZ ct1, ct2;
		ct1.x = centroid1(0);
		ct1.y = centroid1(1);
		ct1.z = centroid1(2);
		
		ct2.x = centroid2(0);
		ct2.y = centroid2(1);
		ct2.z = centroid2(2);

		double a1, a2;
		a1 = angle_bt_vectors( end, ct1, end1);
		a2 = angle_bt_vectors( end, ct2, end1);

		if( a1 > a2)
			seg = c1;
		else
			seg = c2;

		cout<<seg->size()<<endl;
	}
	else
	{
		std::vector<int> vec = indices[0].indices;
		double deviation = 0.08;
		for( int i = 0; i < vec.size(); i++)
			if( search->points[i].y < end.y + deviation && search->points[i].y > end.y - deviation)
				seg->push_back( search->points[ vec[i]]);
	
		// if size of cluster is larger than 2000, it means it attaches to the body
		// that we give a passthrough filter along the axis of body with reasonable deviation
		// to do

		cout<<"only 1 cluster"<<endl;	
		cout<<"size of cluster:"<<vec.size()<<endl;
	}

}

int get_min_sequence(Eigen::EigenSolver<Eigen::MatrixXd>& es)
{
	std::complex<double> me;
	me = es.eigenvalues()[0];
	double min = std::norm( me);
	int i,idx = 0;
	for( i = 1; i < 6; i++)
	{
		me = es.eigenvalues()[i];
		if(min > std::norm( me))
		{
			min = std::norm( me);
			idx = i;
		}
	}
																		    
	return idx;
}

double angle_bt_vectors( pcl::PointXYZ& p1, pcl::PointXYZ& p2, pcl::PointXYZ& p3)
{
	Eigen::Vector3d v1(p2.x-p1.x, p2.y-p1.y, p2.z-p1.z);
	Eigen::Vector3d v2(p3.x-p1.x, p3.y-p1.y, p3.z-p1.z);
	double r = v1.dot(v2);
	r = r/v1.norm()/v2.norm();
	return acos(r);
}

void pose_recognition( pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& in, pcl::PointXYZ& pt, Eigen::Vector3d& vecA)
{
	Eigen::VectorXd th(6);
	th << 0, 1, 1, 0, 0, 0;
	double u = 1, v = 0, w = 0;
	int cloud_size = in->size();

	srand((unsigned)time(NULL));
	pcl::PointXYZRGBA Pi;

	for( int k = 0; k < 20000; k++)
	{
		Eigen::VectorXd phi(6);	  
		double random = (double)(rand()) / (RAND_MAX + 1.0);
		unsigned int index = static_cast<unsigned int>(random*cloud_size);	
		Pi = in->points[index];

		double a, b, c;
		a = Pi.x - pt.x;
		b = Pi.y - pt.y;
		c = Pi.z - pt.z;

		random = (double)(rand()) / (RAND_MAX + 1.0);
		index = static_cast<unsigned int>(random*cloud_size);	
		Pi = in->points[index];

		double a1, b1, c1;
		a1 = Pi.x - pt.x;
		b1 = Pi.y - pt.y;
		c1 = Pi.z - pt.z;
		
		phi(0) = a*a - a1*a1;
		phi(1) = b*b - b1*b1;
		phi(2) = c*c - c1*c1;
		phi(3) = a*b - a1*b1;
		phi(4) = b*c - b1*c1;
		phi(5) = a*c - a1*c1;

		double z_head = th.transpose()*phi;
		double error = -z_head;

		double gamma = 40; 
		Eigen::VectorXd th_dot(6);
		th_dot = gamma*phi*error;
		
		// adaptive law with projection
		for( int i = 0; i < 3; i++)
			if( th(i) < 0 && th_dot(i) < 0)
				th_dot(i) = 0;
				
		// estimated theta update
		th = th + th_dot*0.001;
		// set zero if theta smaller than zero
		for( int i = 0; i < 3; i++)
			if( th(i) < 0)
				th(i) = 0;
		
		// indirect scheme
		if( (fabs(th(3)) < 0.01  && fabs(th(4)) < 0.01) || (fabs(th(3)) < 0.01 && fabs(th(5)) < 0.01) || (fabs(th(4)) < 0.01 && fabs(th(5)) < 0.01)  )
			indirect_scheme1(th, u, v, w);
	    	else
			indirect_scheme( th, u, v, w);
	}

	do{
		double random = (double)(rand()) / (RAND_MAX + 1.0);
		unsigned int index = static_cast<unsigned int>(random*cloud_size);
		Pi = in->points[index];
	}while( Pi.x*Pi.x + Pi.y*Pi.y + Pi.z*Pi.z < 0.04);
	
	pcl::PointXYZ pt1 = pcl::PointXYZ(Pi.x, Pi.y, Pi.z);
    
	if( (fabs(th(3)) < 0.01  && fabs(th(4)) < 0.01) || (fabs(th(3)) < 0.01 && fabs(th(5)) < 0.01) || (fabs(th(4)) < 0.01 && fabs(th(5)) < 0.01)  )
		Direction_Decision1( th, u, v, w, pt, pt1); 
	else
		Direction_Decision( u, v, w, pt, pt1); 

	vecA(0) = u;
	vecA(1) = v;
	vecA(2) = w;
}

void indirect_scheme( Eigen::VectorXd& th, double& u, double& v, double& w)
{
	double beta, alpha;
	beta = atan2(th(4), th(5));
	alpha = atan2(th(3), th(5)*sin(beta));

	u = sin(alpha)*cos(beta);
	v = sin(alpha)*sin(beta);
	w = cos(alpha);
}

void indirect_scheme1( Eigen::VectorXd& th, double& u, double& v, double& w)
{	
	if( th(0) > 1)
		th(0) = 1;
	if( th(1) > 1)
		th(1) = 1;
	if( th(2) > 1)
		th(2) = 1;

	double s = 2.0/(th(0)+th(1)+th(2));
	double t0 = th(0)*s;
	double t1 = th(1)*s;
	double t2 = th(2)*s;

	if( t0 > 1)
		t0 = 1;
	if( t1 > 1)
		t1 = 1;
	if( t2 > 1)
		t2 = 1;

	u = sqrt( 1 - t0);
	v = sqrt( 1 - t1);
	w = sqrt( 1 - t2);
}

void Direction_Decision( double& u, double& v, double& w, pcl::PointXYZ& J0, pcl::PointXYZ& Pi)
{
	pcl::PointXYZ J1, J2;
	J1 = pcl::PointXYZ(u,v,w);
	J2 = pcl::PointXYZ(-u,-v,-w);
	
	double a1, a2;
	a1 = angle_bt_vectors( J0, J1, Pi);		
	a2 = angle_bt_vectors( J0, J2, Pi);		

	if( a1 > a2)
	{
		u = -u;	v = -v; w = -w;
	}
}

void Direction_Decision1( Eigen::VectorXd& th, double& u, double& v, double& w, pcl::PointXYZ& J0, pcl::PointXYZ& Pi)
{
	pcl::PointXYZ J1, J2;
	// determine the direction
	if( fabs( th(3)) < 1e-4)
		th(3) = 0;
	if( fabs( th(4)) < 1e-4)
		th(4) = 0;
	if( fabs( th(5)) < 1e-4)
		th(5) = 0;
	if( th(3) <= 0 && th(4) <= 0 && th(5) <= 0)
	{
		J1 = pcl::PointXYZ(u,v,w);
		J2 = pcl::PointXYZ(-u,-v,-w);

		double a1, a2;
		a1 = angle_bt_vectors( J0, J1, Pi);		
		a2 = angle_bt_vectors( J0, J2, Pi);		

		if( a1 > a2)
		{
			u = -u;	v = -v; w = -w;
		}
	}
	if( th(3) <= 0 && th(4) >= 0 && th(5) >= 0)
	{
		J1 = pcl::PointXYZ(u,v,-w);
		J2 = pcl::PointXYZ(-u,-v,w);
	
		double a1, a2;
		a1 = angle_bt_vectors( J0, J1, Pi);		
		a2 = angle_bt_vectors( J0, J2, Pi);		

		if( a1 > a2)
		{
			u = -u;	v = -v;
		}
		else
			w = -w;		
			
	}
	if( th(3) >= 0 && th(4) >= 0 && th(5) <= 0)
	{
		J1 = pcl::PointXYZ(u,-v,w);
		J2 = pcl::PointXYZ(-u,v,-w);
			double a1, a2;
		a1 = angle_bt_vectors( J0, J1, Pi);		
		a2 = angle_bt_vectors( J0, J2, Pi);		

		if( a1 > a2)
		{
			u = -u;	w = -w;
		}
		else
			v = -v;
	
	}
	if( th(3) >= 0 && th(4) <= 0 && th(5) >= 0)
	{
		J1 = pcl::PointXYZ(u,-v,-w);
		J2 = pcl::PointXYZ(-u,v,w);
			double a1, a2;
		a1 = angle_bt_vectors( J0, J1, Pi);		
		a2 = angle_bt_vectors( J0, J2, Pi);		

		if( a1 > a2)
			u = -u;
		else
		{
			v = -v;	w = -w;
		}
	}
}

void show_visualizer( pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud, HumanPose& pose)
{
	pcl::ModelCoefficients rlowerArm, llowerArm, rupperArm, lupperArm;
	rlowerArm.values.push_back(pose.rHand.x);
	rlowerArm.values.push_back(pose.rHand.y);
	rlowerArm.values.push_back(pose.rHand.z);
	rlowerArm.values.push_back(pose.rElbow.x-pose.rHand.x);
	rlowerArm.values.push_back(pose.rElbow.y-pose.rHand.y - 0.03);
	rlowerArm.values.push_back(pose.rElbow.z-pose.rHand.z);
	rlowerArm.values.push_back(pose.lowerArmRadius);

	llowerArm.values.push_back(pose.lHand.x);
	llowerArm.values.push_back(pose.lHand.y);
	llowerArm.values.push_back(pose.lHand.z);
	llowerArm.values.push_back(pose.lElbow.x-pose.lHand.x);
	llowerArm.values.push_back(pose.lElbow.y-pose.lHand.y);
	llowerArm.values.push_back(pose.lElbow.z-pose.lHand.z);
	llowerArm.values.push_back(pose.lowerArmRadius);

	rupperArm.values.push_back(pose.rElbow.x);
	rupperArm.values.push_back(pose.rElbow.y - 0.03);
	rupperArm.values.push_back(pose.rElbow.z);
	rupperArm.values.push_back(pose.rShoulder.x-pose.rElbow.x);
	rupperArm.values.push_back(pose.rShoulder.y-pose.rElbow.y);
	rupperArm.values.push_back(pose.rShoulder.z-pose.rElbow.z);
	rupperArm.values.push_back(pose.upperArmRadius);
    
	lupperArm.values.push_back(pose.lElbow.x);
	lupperArm.values.push_back(pose.lElbow.y);
	lupperArm.values.push_back(pose.lElbow.z);
	lupperArm.values.push_back(pose.lShoulder.x-pose.lElbow.x);
	lupperArm.values.push_back(pose.lShoulder.y-pose.lElbow.y);
	lupperArm.values.push_back(pose.lShoulder.z-pose.lElbow.z);
	lupperArm.values.push_back(pose.upperArmRadius);
	
	pcl::ModelCoefficients rlowerFeet, llowerFeet, rupperFeet, lupperFeet;
	rlowerFeet.values.push_back(pose.rAnkle.x);
	rlowerFeet.values.push_back(pose.rAnkle.y);
	rlowerFeet.values.push_back(pose.rAnkle.z);
	rlowerFeet.values.push_back(pose.rKnee.x-pose.rAnkle.x);
	rlowerFeet.values.push_back(pose.rKnee.y-pose.rAnkle.y);
	rlowerFeet.values.push_back(pose.rKnee.z-pose.rAnkle.z);
	rlowerFeet.values.push_back(pose.lowerFeetRadius);
	
	llowerFeet.values.push_back(pose.lAnkle.x);
	llowerFeet.values.push_back(pose.lAnkle.y);
	llowerFeet.values.push_back(pose.lAnkle.z);
	llowerFeet.values.push_back(pose.lKnee.x-pose.lAnkle.x);
	llowerFeet.values.push_back(pose.lKnee.y-pose.lAnkle.y);
	llowerFeet.values.push_back(pose.lKnee.z-pose.lAnkle.z);
	llowerFeet.values.push_back(pose.lowerFeetRadius);
	
	rupperFeet.values.push_back(pose.rKnee.x);
	rupperFeet.values.push_back(pose.rKnee.y);
	rupperFeet.values.push_back(pose.rKnee.z);
	rupperFeet.values.push_back(pose.rHip.x-pose.rKnee.x);
	rupperFeet.values.push_back(pose.rHip.y-pose.rKnee.y);
	rupperFeet.values.push_back(pose.rHip.z-pose.rKnee.z);
	rupperFeet.values.push_back(pose.upperFeetRadius);
	
	lupperFeet.values.push_back(pose.lKnee.x);
	lupperFeet.values.push_back(pose.lKnee.y);
	lupperFeet.values.push_back(pose.lKnee.z);
	lupperFeet.values.push_back(pose.lHip.x-pose.lKnee.x);
	lupperFeet.values.push_back(pose.lHip.y-pose.lKnee.y);
	lupperFeet.values.push_back(pose.lHip.z-pose.lKnee.z);
	lupperFeet.values.push_back(pose.upperFeetRadius);
	
	pcl::visualization::PCLVisualizer viewer ("Viewer");
	viewer.setBackgroundColor (0.2, 0.2, 0.2);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgba(cloud);
	viewer.addPointCloud<pcl::PointXYZRGBA> (cloud, rgba, "cloud");
	viewer.addCylinder(rlowerArm, "rlowerArm");
	viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 1.0, 0.0, "rlowerArm");
	viewer.addCylinder(llowerArm, "llowerArm");
	viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 1.0, 0.0, "llowerArm");	
	viewer.addCylinder(rlowerFeet, "rlowerFeet");
	viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 1.0, "rlowerFeet");
	viewer.addCylinder(llowerFeet, "llowerFeet");
	viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 1.0, "llowerFeet");
	viewer.addCylinder(rupperArm, "rupperArm");
	viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 0.0, "rupperArm");
	viewer.addCylinder(lupperArm, "lupperArm");
	viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 0.0, "lupperArm");
	viewer.addCylinder(rupperFeet, "rupperFeet");
	viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "rupperFeet");
	viewer.addCylinder(lupperFeet, "lupperFeet");
	viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "lupperFeet");
		
	while (!viewer.wasStopped ())
    {
		viewer.spinOnce (100);
		boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }
}

void   show_visualizer_points( pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud, HumanPose& pose)
{
	pcl::visualization::PCLVisualizer viewer ("Viewer - Joints");
	viewer.setBackgroundColor (0.2, 0.2, 0.2);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgba(cloud);
	viewer.addPointCloud<pcl::PointXYZRGBA> (cloud, rgba, "cloud");
	viewer.addSphere( pose.rHand, 0.05, 1, 1, 0, "rHand");
	viewer.addSphere( pose.lHand, 0.05, 1, 1, 0, "lHand");
	viewer.addSphere( pose.rElbow, 0.05, 1, 1, 0, "rElbow");
	viewer.addSphere( pose.lElbow, 0.05, 1, 1, 0, "lElbow");
	viewer.addSphere( pose.rShoulder, 0.05, 1, 1, 0, "rShoulder");
	viewer.addSphere( pose.lShoulder, 0.05, 1, 1, 0, "lShoulder");
	viewer.addSphere( pose.rAnkle, 0.05, 1, 1, 0, "rAnkle");
	viewer.addSphere( pose.lAnkle, 0.05, 1, 1, 0, "lAnkle");
	viewer.addSphere( pose.rKnee, 0.05, 1, 1, 0, "rKnee");
	viewer.addSphere( pose.lKnee, 0.05, 1, 1, 0, "lKnee");
	viewer.addSphere( pose.rHip, 0.05, 1, 1, 0, "rHip");
	viewer.addSphere( pose.lHip, 0.05, 1, 1, 0, "lHip");
		
	while (!viewer.wasStopped ())
    {
		viewer.spinOnce (100);
		boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }
}


