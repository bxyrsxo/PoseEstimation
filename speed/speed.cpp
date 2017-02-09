#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/ModelCoefficients.h>

#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/project_inliers.h>

void Data_Generation( pcl::PointCloud<pcl::PointXYZ>::Ptr& pc, double r);
void Cloud_Transform( pcl::PointCloud<pcl::PointXYZ>::Ptr& in, pcl::PointCloud<pcl::PointXYZ>::Ptr& out, Eigen::Matrix3d& R);
void Adaptive_Estimation( pcl::PointCloud<pcl::PointXYZ>::Ptr& pc, Eigen::Matrix3d& R);
Eigen::Matrix3d rotation_matrix( double a, double b, double c);
void Ransac_Estimation( pcl::PointCloud<pcl::PointXYZ>::Ptr& pc, pcl::ModelCoefficients::Ptr& coeff);

// time comsuption
double total_time = 0.0;
double gain = 0.0;

fstream f;

int
main (int argc, char** argv)
{
	if( argc != 2)
	{
		cout<<"three arguments: .exe [radius] [gain]"<<endl;
		return 0;
	}
	// generate the point cloud--------------------------------------------------------
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
	Data_Generation( cloud, atof( argv[1]));

	// cloud transform
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_trans (new pcl::PointCloud<pcl::PointXYZ>);
	srand((unsigned)time(NULL));
	
	Eigen::Matrix3d R;
	const int times = 100;
	double search_step = 0.0001;
	f.open("gain_time.txt", ios::out);

for( double j = 0.01; j > 0.00001; j -= search_step)
{
	gain = j;
	pcl::ScopeTime t ("Adaptive Estimation");
	{
		for( int i = 0; i < times; i++)
		{
			double a = (double)(rand()) / (RAND_MAX + 1.0) * 360;
			double b = (double)(rand()) / (RAND_MAX + 1.0) * 360;
			double c = (double)(rand()) / (RAND_MAX + 1.0) * 360;
			R = rotation_matrix(a, b, c);
			Cloud_Transform( cloud, cloud_trans, R); 
			Adaptive_Estimation( cloud_trans, R);
		}
	}	
//	if( fabs(j-0.01) < 1e-9 || fabs(j-0.001) < 1e-9)
//		search_step /= 10.0;
	f<<gain<<" "<<t.getTime()<<endl;
}
	f.close();

	return 0;
}

void Data_Generation( pcl::PointCloud<pcl::PointXYZ>::Ptr& pc, double r)
{
	pcl::ScopeTime t ("Data Generation");

	for (float z(0.0); z <= 30; z += 0.5)
		for (float angle(0.0); angle <= 360.0; angle += 5.0)
		{
			pcl::PointXYZ point;
			point.x = r * cosf (pcl::deg2rad(angle));
			point.y = r * sinf (pcl::deg2rad(angle));
			point.z = z;
			pc->points.push_back(point);
		}
}

void Cloud_Transform( pcl::PointCloud<pcl::PointXYZ>::Ptr& in, pcl::PointCloud<pcl::PointXYZ>::Ptr& out, Eigen::Matrix3d& R)
{
	// input data transformation
	Eigen::Vector3d pi_vec, p_vec;
	out->clear();
	for( int i = 0; i < in->size(); i++)
	{
		pi_vec(0) = in->points[i].x;
		pi_vec(1) = in->points[i].y;
		pi_vec(2) = in->points[i].z;
		
		p_vec = R*pi_vec;
		out->push_back( pcl::PointXYZ( p_vec(0), p_vec(1), p_vec(2)));
	}
}

void Adaptive_Estimation( pcl::PointCloud<pcl::PointXYZ>::Ptr& pc, Eigen::Matrix3d& R)
{
//	pcl::ScopeTime t ("Adaptive Estimation");
	Eigen::VectorXd th(6);
	th << 1, 0, 1, 0, 0, 0;
	
	Eigen::VectorXd phi(6);
	Eigen::VectorXd th_dot(6);
	
	double u = 0, v = 0, w = 1;
	int cloud_size = pc->size();

	pcl::PointXYZ J0(0.0,0.0,0.0);
	double sampling_time = 0.001;
	double gamma = 5;

	Eigen::Vector3d opt;
	opt(0) = R(0,0)*u + R(0,1)*v + R(0,2)*w;
	opt(1) = R(1,0)*u + R(1,1)*v + R(1,2)*w;
	opt(2) = R(2,0)*u + R(2,1)*v + R(2,2)*w;

	Eigen::VectorXd th_opt(6), err(6);
	th_opt<<opt(1)*opt(1)+opt(2)*opt(2),
		    opt(0)*opt(0)+opt(2)*opt(2),
			opt(0)*opt(0)+opt(1)*opt(1),
		    -2*opt(0)*opt(1),
			-2*opt(1)*opt(2),
			-2*opt(0)*opt(2);


    pcl::PointXYZ Pi;
	int i, k;
	double a, b, c, a1, b1, c1;
	double rms;
	const double threshold = 1e-2;
	int breakcount = 0;
	int loop_times = 20000;
	bool have_broken = 0;
	unsigned int index;
	double random;
	double z, z_head, error;

	for( k = 0; k < loop_times; k++)
	{
		random = (double)(rand()) / (RAND_MAX + 1.0);
		index = static_cast<unsigned int>(random*cloud_size);
		Pi = pc->points[index];
		
		a = Pi.x - J0.x;
		b = Pi.y - J0.y;
		c = Pi.z - J0.z;
		
		random = (double)(rand()) / (RAND_MAX + 1.0);
		index = static_cast<unsigned int>(random*cloud_size);
		Pi = pc->points[index];
		
		a1 = Pi.x - J0.x;
		b1 = Pi.y - J0.y;
		c1 = Pi.z - J0.z;

		phi(0) = a*a - a1*a1;
		phi(1) = b*b - b1*b1;
		phi(2) = c*c - c1*c1;
		phi(3) = a*b - a1*b1;
		phi(4) = b*c - b1*c1;
		phi(5) = a*c - a1*c1;
		
		z_head = th.transpose()*phi;
		error = -z_head;
		
		th_dot = gain*phi*error;
		// adaptive law with projection
		for( i = 0; i < 3; i++)
			if( th(i) < 0 && th_dot(i) < 0)
				th_dot(i) = 0;
				
		// estimated theta update
		th = th + th_dot*sampling_time;
	
		double s = 2.0 / (th(0) + th(1) + th(2));

		double err_arr[3];
		err_arr[0] = s*th(0) - th_opt(0);
		err_arr[1] = s*th(1) - th_opt(1);
		err_arr[2] = s*th(2) - th_opt(2);

		double tmp = sqrt( err_arr[0]*err_arr[0] + err_arr[1]*err_arr[1] + err_arr[2]*err_arr[2]);
		
		if ( tmp < threshold)
			breakcount++;	
		
		if( breakcount == 100 && have_broken == 0)
			have_broken = 1;
		
		if( have_broken)
			break;
	}
	
//	total_time += t.getTime();
}

Eigen::Matrix3d rotation_matrix( double a, double b, double c)
{
	Eigen::Matrix3d Rz, Ry, Rx, R;
	Rz << cos(a), -sin(a), 0, sin(a),  cos(a), 0, 0, 0, 1;
	Ry << cos(b), 0, sin(b), 0, 1, 0, -sin(b), 0, cos(b);
	Rx << 1, 0, 0, 0, cos(c), -sin(c), 0, sin(c), cos(c); 
	R = Rz*Ry*Rx;
	return R;
}

void Ransac_Estimation( pcl::PointCloud<pcl::PointXYZ>::Ptr& pc, pcl::ModelCoefficients::Ptr& coeff)
{
	pcl::ScopeTime t ("RANSAC Estimation");
 
    // RANSAC estimation
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;
    seg.setOptimizeCoefficients (true);

    seg.setModelType( pcl::SACMODEL_CYLINDER);
    seg.setMethodType( pcl::SAC_RANSAC);
    seg.setDistanceThreshold( 0.01);

	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
	// Estimate point normals
	ne.setSearchMethod (tree);
	ne.setInputCloud (pc);
	ne.setKSearch (50);
	ne.compute (*normals);

    seg.setInputCloud( pc->makeShared());
	seg.setInputNormals( normals);
	seg.setNormalDistanceWeight (0.1);
    seg.setMaxIterations (500);
	seg.setDistanceThreshold (0.25);
	seg.setRadiusLimits (1, 1);
    seg.segment(*inliers, *coeff);
}
