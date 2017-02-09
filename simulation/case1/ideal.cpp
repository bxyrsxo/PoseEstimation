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

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

void Data_Generation( pcl::PointCloud<pcl::PointXYZ>::Ptr& pc);
void white_noise( pcl::PointCloud<pcl::PointXYZ>::Ptr& pc);
void partial_cloud( pcl::PointCloud<pcl::PointXYZ>::Ptr& pc);
void Cloud_Transform( pcl::PointCloud<pcl::PointXYZ>::Ptr& in, pcl::PointCloud<pcl::PointXYZ>::Ptr& out, Eigen::Matrix3d& R);
void Adaptive_Estimation( pcl::PointCloud<pcl::PointXYZ>::Ptr& pc, Eigen::Matrix3d& R);
void indirect_scheme ( Eigen::VectorXd& th, double& u, double& v, double& w);
void indirect_scheme1( Eigen::VectorXd& th, double& u, double& v, double& w);
void Direction_Decision ( double& u, double& v, double& w, pcl::PointXYZ& J0, pcl::PointXYZ& Pi);
void Direction_Decision1( Eigen::VectorXd& th, double& u, double& v, double& w, pcl::PointXYZ& J0, pcl::PointXYZ& Pi);
double angle_bt_vectors( pcl::PointXYZ& p1, pcl::PointXYZ& p2, pcl::PointXYZ& p3);
Eigen::Matrix3d rotation_matrix( double a, double b, double c);

double err_IAC = 0.0;
double err_IAS = 0.0;
int err_cnt = 0;

int
main (int argc, char** argv)
{
	// generate the point cloud--------------------------------------------------------
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
	Data_Generation( cloud);

	// partial cloud
	partial_cloud( cloud);

	// add white noise
	// white_noise( cloud);

	srand((unsigned)time(NULL));
	Eigen::Matrix3d R;
	// cloud transform
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_trans (new pcl::PointCloud<pcl::PointXYZ>);
	double a, b, c;

	for( int i = 0; i < 100; i++)
	{
		a = (double)(rand()) / (RAND_MAX + 1.0) * 360;
		b = (double)(rand()) / (RAND_MAX + 1.0) * 360;
		c = (double)(rand()) / (RAND_MAX + 1.0) * 360;
	    R = rotation_matrix(a, b, c);
		Cloud_Transform( cloud, cloud_trans, R); 
		Adaptive_Estimation( cloud_trans, R);
	}

	std::cout<<"error IAC:"<<err_IAC/err_cnt<<endl;
	std::cout<<"error IAS:"<<err_IAS/err_cnt<<endl;

	return 0;
}

void Data_Generation( pcl::PointCloud<pcl::PointXYZ>::Ptr& pc)
{
	pcl::ScopeTime t ("Data Generation");
	for (float z(0.0); z <= 5.0; z += 0.25)
	{
		static int q = 1;	
		for (float angle(0.0); angle <= 180.0; angle += 5.0)
		{
			pcl::PointXYZ point;
			point.x = (1.0+0.025*q)*cosf (pcl::deg2rad(angle));
			point.y = (1.0+0.025*q)*sinf (pcl::deg2rad(angle));
			point.z = z;
			pc->points.push_back(point);
		}
		q++;
	}
}

void Adaptive_Estimation( pcl::PointCloud<pcl::PointXYZ>::Ptr& pc, Eigen::Matrix3d& R)
{
	// pcl::ScopeTime t ("Adaptive Estimation");
	const double radius = 1.0;
	const double L = 5.0;

	Eigen::VectorXd th(6);
	th << 1, 0, 1, 0, 0, 0;
	double u = 0, v = 0, w = 1;
	double u1, v1, w1;
	int cloud_size = pc->size();

	pcl::PointXYZ J0(0.0,0.0,0.0);
	double sampling_time = 0.001;
	double gamma = 5;

    pcl::PointXYZ Pi;
	int k;
	for( k = 0; k < 20000; k++)
	{
		Eigen::VectorXd phi(6);
		do{
			double random = (double)(rand()) / (RAND_MAX + 1.0);
			unsigned int index = static_cast<unsigned int>(random*cloud_size);
			Pi = pc->points[index];
		}while( Pi.x*Pi.x + Pi.y*Pi.y + Pi.z*Pi.z < 2);
		
		double a, b, c;
		a = Pi.x - J0.x;
		b = Pi.y - J0.y;
		c = Pi.z - J0.z;
	
		do{
			double random = (double)(rand()) / (RAND_MAX + 1.0);
			unsigned int index = static_cast<unsigned int>(random*cloud_size);
			Pi = pc->points[index];
		}while( Pi.x*Pi.x + Pi.y*Pi.y + Pi.z*Pi.z < 2);

		double a1, b1, c1;
		a1 = Pi.x - J0.x;
		b1 = Pi.y - J0.y;
		c1 = Pi.z - J0.z;
		
		phi(0) = a*a - a1*a1;
		phi(1) = b*b - b1*b1;
		phi(2) = c*c - c1*c1;
		phi(3) = a*b - a1*b1;
		phi(4) = b*c - b1*c1;
		phi(5) = a*c - a1*c1;
		
		double z_head = th.transpose()*phi;
		double error = -z_head;
		
		Eigen::VectorXd th_dot(6);
		th_dot = gamma*phi*error;
		// adaptive law with projection
		for( int i = 0; i < 3; i++)
			if( th(i) < 0 && th_dot(i) < 0)
				th_dot(i) = 0;
				
		// estimated theta update
		th = th + th_dot*sampling_time;
		// set zero if theta smaller than zero
		for( int i = 0; i < 3; i++)
			if( th(i) < 0)
				th(i) = 0;


		indirect_scheme( th, u, v, w);
		Direction_Decision( u, v, w, J0, Pi);
	
		indirect_scheme1(th, u1, v1, w1);
		Direction_Decision1( th, u1, v1, w1, J0, Pi);
	}

	double tmp = sqrt((R(0,2)-u)*(R(0,2)-u) + (R(1,2)-v)*(R(1,2)-v) + (R(2,2)-w)*(R(2,2)-w));
	double tmp1 = sqrt((R(0,2)-u1)*(R(0,2)-u1) + (R(1,2)-v1)*(R(1,2)-v1) + (R(2,2)-w1)*(R(2,2)-w1));

	if( tmp < 1 || tmp1 < 1)
	{
		err_IAC += tmp1;
		err_IAS += tmp;
		err_cnt ++;
	}


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

double angle_bt_vectors( pcl::PointXYZ& p1, pcl::PointXYZ& p2, pcl::PointXYZ& p3)
{
	Eigen::Vector3d v1(p2.x-p1.x, p2.y-p1.y, p2.z-p1.z);
	Eigen::Vector3d v2(p3.x-p1.x, p3.y-p1.y, p3.z-p1.z);
	double r = v1.dot(v2);
	r = r/v1.norm()/v2.norm();
	return acos(r);
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

void white_noise( pcl::PointCloud<pcl::PointXYZ>::Ptr& pc)
{
	boost::mt19937 rng; // I don't seed it on purpouse (it's not relevant)
	rng.seed(time(NULL));
	
	// simulate rolling a die
	boost::normal_distribution<> nd(0.0, 0.02);
	boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_nor(rng, nd);
	
	for( int i = 0; i < pc->size(); i++)
	{
		pc->points[i].x +=  var_nor();	
		pc->points[i].y +=  var_nor();	
		pc->points[i].z +=  var_nor();	
	}
	
}

void partial_cloud( pcl::PointCloud<pcl::PointXYZ>::Ptr& pc)
{
	srand( (unsigned) time(NULL));	
	int cloud_size = pc->size();
	
	pcl::PointCloud<pcl::PointXYZ>::Ptr cp (new pcl::PointCloud<pcl::PointXYZ>); 
	pcl::copyPointCloud( *pc, *cp);
	pc->clear();

	int num_points = cloud_size / 8;
	
	for( int i = 0; i < num_points; i++)
	{	
		double random = (double)(rand()) / (RAND_MAX + 1.0);
		unsigned int index = static_cast<unsigned int>(random*cloud_size);
		pc->push_back( cp->points[index]);
	}
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

void Direction_Decision1( Eigen::VectorXd& th, double& u, double& v, double& w, pcl::PointXYZ& J0, pcl::PointXYZ& Pi)
{
	pcl::PointXYZ J1, J2, J3, J4;
	// determine the direction
	if( fabs( th(3)) < 1e-4)
		th(3) = 0;
	if( fabs( th(4)) < 1e-4)
		th(4) = 0;
	if( fabs( th(5)) < 1e-4)
		th(5) = 0;

	bool on_plane = 0;
	if( th(3) == 0 && th(4) == 0)
		on_plane = 1;
	if( th(3) == 0 && th(5) == 0)
		on_plane = 2;
	if( th(4) == 0 && th(5) == 0)
		on_plane = 3;
	
	if( !on_plane)
	{
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
	else
	{
		if( on_plane == 1)    //  u,w have the same sign
		{
			if( th(5) < 0)
			{
				J1 = pcl::PointXYZ( u, v, w);
				J2 = pcl::PointXYZ( u,-v, w);
				J3 = pcl::PointXYZ(-u, v,-w);
				J4 = pcl::PointXYZ(-u,-v,-w);

				double a[4];
				a[0] = angle_bt_vectors( J0, J1, Pi);		
				a[1] = angle_bt_vectors( J0, J2, Pi);		
				a[2] = angle_bt_vectors( J0, J3, Pi);		
				a[3] = angle_bt_vectors( J0, J4, Pi);		

				int smallest = 0;
				double smallest_value = a[0];
				for( int i = 1; i < 4; i++)
					if( a[i] < smallest_value)
					{
						smallest = i;
						smallest_value = a[i];
					}

				switch(smallest)
				{
					case 1:
						v = -v;					
						break;
					case 2:
						u = -u; w = -w;
						break;
					case 3:
						u = -u; v = -v; w = -w;
						break;
				}
			}
			else
			{
				J1 = pcl::PointXYZ( u, v,-w);
				J2 = pcl::PointXYZ( u,-v,-w);
				J3 = pcl::PointXYZ(-u, v, w);
				J4 = pcl::PointXYZ(-u,-v, w);

				double a[4];
				a[0] = angle_bt_vectors( J0, J1, Pi);		
				a[1] = angle_bt_vectors( J0, J2, Pi);		
				a[2] = angle_bt_vectors( J0, J3, Pi);		
				a[3] = angle_bt_vectors( J0, J4, Pi);		

				int smallest = 0;
				double smallest_value = a[0];
				for( int i = 1; i < 4; i++)
					if( a[i] < smallest_value)
					{
						smallest = i;
						smallest_value = a[i];
					}

				switch(smallest)
				{
					case 0:
						w = -w;
						break;
					case 1:
						v = -v; w = -w;					
						break;
					case 2:
						u = -u; 
						break;
					case 3:
						u = -u; v = -v;
						break;
				}
			}
		}
		if( on_plane == 2)
		{
			if( th(4) < 0)
			{
				J1 = pcl::PointXYZ( u, v, w);
				J2 = pcl::PointXYZ(-u, v, w);
				J3 = pcl::PointXYZ( u,-v,-w);
				J4 = pcl::PointXYZ(-u,-v,-w);

				double a[4];
				a[0] = angle_bt_vectors( J0, J1, Pi);		
				a[1] = angle_bt_vectors( J0, J2, Pi);		
				a[2] = angle_bt_vectors( J0, J3, Pi);		
				a[3] = angle_bt_vectors( J0, J4, Pi);		

				int smallest = 0;
				double smallest_value = a[0];
				for( int i = 1; i < 4; i++)
					if( a[i] < smallest_value)
					{
						smallest = i;
						smallest_value = a[i];
					}

				switch(smallest)
				{
					case 1:
						u = -u;					
						break;
					case 2:
						v = -v; w = -w;
						break;
					case 3:
						u = -u; v = -v; w = -w;
						break;
				}
			}
			else
			{
				J1 = pcl::PointXYZ( u,-v, w);
				J2 = pcl::PointXYZ(-u,-v, w);
				J3 = pcl::PointXYZ( u, v,-w);
				J4 = pcl::PointXYZ(-u, v,-w);

				double a[4];
				a[0] = angle_bt_vectors( J0, J1, Pi);		
				a[1] = angle_bt_vectors( J0, J2, Pi);		
				a[2] = angle_bt_vectors( J0, J3, Pi);		
				a[3] = angle_bt_vectors( J0, J4, Pi);		

				int smallest = 0;
				double smallest_value = a[0];
				for( int i = 1; i < 4; i++)
					if( a[i] < smallest_value)
					{
						smallest = i;
						smallest_value = a[i];
					}

				switch(smallest)
				{
					case 0:
						v = -v;
						break;
					case 1:
						u = -u; v = -v;					
						break;
					case 2:
						w = -w; 
						break;
					case 3:
						u = -u; w = -w;
						break;
				}
			}
		}
		if( on_plane == 3)
		{
			if( th(3) < 0)
			{
				J1 = pcl::PointXYZ( u, v, w);
				J2 = pcl::PointXYZ( u, v,-w);
				J3 = pcl::PointXYZ(-u,-v, w);
				J4 = pcl::PointXYZ(-u,-v,-w);

				double a[4];
				a[0] = angle_bt_vectors( J0, J1, Pi);		
				a[1] = angle_bt_vectors( J0, J2, Pi);		
				a[2] = angle_bt_vectors( J0, J3, Pi);		
				a[3] = angle_bt_vectors( J0, J4, Pi);		

				int smallest = 0;
				double smallest_value = a[0];
				for( int i = 1; i < 4; i++)
					if( a[i] < smallest_value)
					{
						smallest = i;
						smallest_value = a[i];
					}

				switch(smallest)
				{
					case 1:
						w = -w;					
						break;
					case 2:
						u = -u; v = -v;
						break;
					case 3:
						u = -u; v = -v; w = -w;
						break;
				}
			}
			else
			{
				J1 = pcl::PointXYZ( u,-v, w);
				J2 = pcl::PointXYZ( u,-v,-w);
				J3 = pcl::PointXYZ(-u, v, w);
				J4 = pcl::PointXYZ(-u, v,-w);

				double a[4];
				a[0] = angle_bt_vectors( J0, J1, Pi);		
				a[1] = angle_bt_vectors( J0, J2, Pi);		
				a[2] = angle_bt_vectors( J0, J3, Pi);		
				a[3] = angle_bt_vectors( J0, J4, Pi);		

				int smallest = 0;
				double smallest_value = a[0];
				for( int i = 1; i < 4; i++)
					if( a[i] < smallest_value)
					{
						smallest = i;
						smallest_value = a[i];
					}

				switch(smallest)
				{
					case 0:
						v = -v;
						break;
					case 1:
						v = -v; w = -w;					
						break;
					case 2:
						u = -u; 
						break;
					case 3:
						u = -u; w = -w;
						break;
				}
			}
		}
	}
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
