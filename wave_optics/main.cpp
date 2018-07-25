#include <opencv2/opencv.hpp>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#define _RESOLUTION_X 1024
#define _RESOLUTION_Y 1024
#define RESOLUTION(X) _RESOLUTION_##X
#define SCALE_FACTOR 4
#define BEND_FACTOR 4000.0f
//#define ANTIALIAS_SHIFT 8
//#define ANTIALIAS_FACTOR (1<<ANTIALIAS_SHIFT)
#define DOWNSAMPLE_FACTOR 4

typedef float (*nfunc)(int,int,void*);
using namespace cv;
using namespace tbb;
int qfill_comp(void *thunk,const void *l,const void *r){
        Point2f *pts = (Point2f *)thunk;
        int il = *(int *)l;
        int ir = *(int *)r;
        if(pts[il].y>pts[ir].y)
                return 1;
        else if(pts[il].y==pts[ir].y)
                return 0;
        else
                return -1;
}
#define SLOPE(p1,p2) (((p1).x-(p2).x)/((p1).y-(p2).y))
#define INTERS(p,k) ((p).x-(k)*(p).y)
#define CLIP(x) ((x)<0?0:((x)<_RESOLUTION_X?(x):(_RESOLUTION_X-1)))
#define YCLIP(x) ((x)<0?0:((x)<_RESOLUTION_Y?(x):(_RESOLUTION_Y-1)))
void qfillTriangle(void* data,void *val,size_t elesize,Point2f *pts){
        int idx[3]={0,1,2};
        qsort_r(idx, 3, sizeof(int), pts, qfill_comp);
        int flag = 0;
        if(pts[idx[1]].y==pts[idx[0]].y){
                float lk = SLOPE(pts[idx[1]],pts[idx[2]]);
                float rk = SLOPE(pts[idx[0]],pts[idx[2]]);
                if(rk > lk){
                        float tmp = lk;
                        lk = rk;
                        rk = tmp;
                }
                float lb = INTERS(pts[idx[2]],lk);
                float rb = INTERS(pts[idx[2]],rk);
                for(int row = ceil(pts[idx[0]].y);row < YCLIP(floor(pts[idx[2]].y))+1;row++){
                        for(size_t ptr = elesize*(row*_RESOLUTION_X+CLIP(ceil(lb+lk*row)));ptr<elesize*(row*_RESOLUTION_X+CLIP(floor(rb+rk*row))+1);ptr+=elesize){
                                memcpy((char *)data+ptr, val, elesize);
                        }
                }
                return;
        }
        float lk = SLOPE(pts[idx[1]],pts[idx[0]]);
        float rk = SLOPE(pts[idx[2]],pts[idx[0]]);
        if(rk < lk){
                float tmp = lk;
                lk = rk;
                rk = tmp;
                flag = 1;
        }
        float lb = INTERS(pts[idx[0]],lk);
        float rb = INTERS(pts[idx[0]],rk);
        for(int row = ceil(pts[idx[0]].y);row < YCLIP(floor(pts[idx[2]].y))+1;row++){
                if(row>pts[idx[1]].y){
                        if(!flag){
                                lk = SLOPE(pts[idx[1]],pts[idx[2]]);
                                lb = INTERS(pts[idx[2]], lk);
                        }
                        else{
                                rk = SLOPE(pts[idx[1]],pts[idx[2]]);
                                rb = INTERS(pts[idx[2]], rk);
                        }
                }
                for(size_t ptr = elesize*(row*_RESOLUTION_X+CLIP(ceil(lb+lk*row)));ptr<elesize*(row*_RESOLUTION_X+CLIP(floor(rb+rk*row))+1);ptr+=elesize){
                        memcpy((char *)data+ptr, val, elesize);
                }
        }
}
void qfillQuad(void* data,void *val,size_t elesize,Point2f *pts){
        qfillTriangle(data, val, elesize, pts);
        qfillTriangle(data, val, elesize, pts+2);
}
void cfillQuad(Mat& buf,unsigned char *color,Point2f *pts){
        Point ipts[4];
#define TPOINT(i) ipts[i] = Point(pts[i].x*8,pts[i].y*8)
        TPOINT(0);
        TPOINT(1);
        TPOINT(2);
        TPOINT(3);
        fillConvexPoly(buf, ipts, 4, Scalar(color[0],color[1],color[2]), CV_AA, 3);
}
void render(nfunc hf,void* args,VideoWriter &outputVideo){
        float *h = (float *)malloc(sizeof(float)*(_RESOLUTION_X/DOWNSAMPLE_FACTOR+3)*(_RESOLUTION_Y/DOWNSAMPLE_FACTOR+3));
        printf("Renderer launched.\n");
        for(int j=0;j<_RESOLUTION_Y/DOWNSAMPLE_FACTOR+3;j++){
                for(int i=0;i<_RESOLUTION_X/DOWNSAMPLE_FACTOR+3;i++){
                        h[i+j*(_RESOLUTION_X/DOWNSAMPLE_FACTOR+3)]=hf(i*DOWNSAMPLE_FACTOR,j*DOWNSAMPLE_FACTOR,args);
                        //printf("x %d y %d h %f\n",i,j,h[i+j*(_RESOLUTION_X/DOWNSAMPLE_FACTOR+3)]);
                }
        }
        float *dx = (float *)malloc(sizeof(float)*(_RESOLUTION_X/DOWNSAMPLE_FACTOR+1)*(_RESOLUTION_Y/DOWNSAMPLE_FACTOR+1));
        float *dy = (float *)malloc(sizeof(float)*(_RESOLUTION_X/DOWNSAMPLE_FACTOR+1)*(_RESOLUTION_Y/DOWNSAMPLE_FACTOR+1));
        for(int j=0;j<_RESOLUTION_Y/DOWNSAMPLE_FACTOR+1;j++){
                for(int i=0;i<_RESOLUTION_X/DOWNSAMPLE_FACTOR+1;i++){
                        dx[i+j*(_RESOLUTION_X/DOWNSAMPLE_FACTOR+1)]=h[i+2+(j+1)*(_RESOLUTION_X/DOWNSAMPLE_FACTOR+3)]-h[i+(j+1)*(_RESOLUTION_X/DOWNSAMPLE_FACTOR+3)];
                        //printf("dx %f\n",dx[i+j*(_RESOLUTION_X/DOWNSAMPLE_FACTOR+1)]);
                        dy[i+j*(_RESOLUTION_X/DOWNSAMPLE_FACTOR+1)]=h[i+1+(j+2)*(_RESOLUTION_X/DOWNSAMPLE_FACTOR+3)]-h[i+1+j*(_RESOLUTION_X/DOWNSAMPLE_FACTOR+3)];
                }
        }
        free(h);
        //Mat res(_RESOLUTION_X,_RESOLUTION_Y,CV_8UC3,Scalar(0,0,0));

        printf("Processing polygons.\n");
#define DPOINT(n,di,dj) pts[n]=Point2f((i+di)*DOWNSAMPLE_FACTOR+dx[i+di+(j+dj)*(_RESOLUTION_X/DOWNSAMPLE_FACTOR+1)]*BEND_FACTOR/DOWNSAMPLE_FACTOR*SCALE_FACTOR*SCALE_FACTOR,(j+dj)*DOWNSAMPLE_FACTOR+dy[i+di+(j+dj)*(_RESOLUTION_X/DOWNSAMPLE_FACTOR+1)]*BEND_FACTOR/DOWNSAMPLE_FACTOR*SCALE_FACTOR*SCALE_FACTOR)
        Mat des = parallel_reduce(blocked_range<int>(0,_RESOLUTION_Y/DOWNSAMPLE_FACTOR),Mat(_RESOLUTION_X,_RESOLUTION_Y,CV_8UC3,Scalar(0,0,0)),[dx,dy](blocked_range<int> &blk,Mat init)->Mat{
                                                                                                                                                       Mat buf(_RESOLUTION_X,_RESOLUTION_Y,CV_8UC3,Scalar(0,0,0));
                                                                                                                                                       Mat res(init);
                                                                                                                                                       for(int j=blk.begin();j<blk.end();j++){
                                                                                                                                                               for(int i=0;i<_RESOLUTION_X/DOWNSAMPLE_FACTOR;i++){
                                                                                                                                                                       Point2f pts[5];
                                                                                                                                                                       DPOINT(0, 0, 0);
                                                                                                                                                                       DPOINT(1, 0, 1);
                                                                                                                                                                       DPOINT(2, 1, 1);
                                                                                                                                                                       DPOINT(3, 1, 0);
                                                                                                                                                                       pts[4] = pts[0];
                                                                                                                                                                       int area2x = (abs((pts[2].y-pts[0].y)*(pts[3].x-pts[1].x)-(pts[2].x-pts[0].x)*(pts[3].y-pts[1].y)));
//printf("area2x %d\n",area2x);
                                                                                                                                                                       if(!area2x)
                                                                                                                                                                               continue;
                                                                                                                                                                       float density = 2*DOWNSAMPLE_FACTOR*DOWNSAMPLE_FACTOR/((float)area2x);
                                                                                                                                                                       //printf("density %f\n",density);
#define CCLIP(x) (((x)<256)?(x):255)
                                                                                                                                                                       unsigned char color[3]={(unsigned char)CCLIP(10*density),(unsigned char)CCLIP(10*((float)j)/(_RESOLUTION_Y/DOWNSAMPLE_FACTOR)*density),(unsigned char)CCLIP(10*((float)i)/(_RESOLUTION_X/DOWNSAMPLE_FACTOR)*density)};
                                                                                                                                                                       //unsigned char color[3]={200*density,200*density,200*density};
                                                                                                                                                                       qfillQuad(buf.data, color, sizeof(unsigned char)*3, pts);
                                                                                                                                                                       //cfillQuad(buf, color, pts);
                                                                                                                                                                       //fillConvexPoly(buf, pts, 4, Scalar(20+20*((float)j)/(_RESOLUTION_Y/DOWNSAMPLE_FACTOR)+20*((float)i)/(_RESOLUTION_X/DOWNSAMPLE_FACTOR),20*((float)j)/(_RESOLUTION_Y/DOWNSAMPLE_FACTOR),20*((float)i)/(_RESOLUTION_X/DOWNSAMPLE_FACTOR))*density, CV_AA,ANTIALIAS_SHIFT);
                                                                                                                                                                       res += buf;
                                                                                                                                                                       buf = Scalar(0,0,0);
                                                                                                                                                               }
                                                                                                                                                       }

                                                                                                                                                       return res;
                                                                                                                                               },[](Mat x,Mat y)->Mat{return x+y;});

        /*
          Point2f pts[5];
          pts[0] = Point2f(0,0);
          pts[1] = Point2f(100,0);
          pts[2] = Point2f(100,100);
          pts[3] = Point2f(0,100);
          pts[4] = Point2f(0,0);
          unsigned char color[3]={255,255,255};
          qfillQuad(buf.data,color,sizeof(unsigned char)*3,pts);
          des+=buf;
          des+=buf;*/
        //medianBlur(des, des, 7);
        printf("Post processing...\n");
        medianBlur(des,des , 13);

        free(dx);
        free(dy);
        //imwrite("result.jpg", des);
        outputVideo.write(des);
}
#define PERIOD 2400
#define SQRT2 1.414213562373095
#define SQRT3 1.732050807568877
float swave(int x,int y,void *args){
        float frame = (float)(*(int *)args);
        return 0.2*cos(2*M_PI*((float)x)/_RESOLUTION_X)*cos(2*M_PI*((float)y)/_RESOLUTION_Y)*sin(2*2*SQRT2*M_PI*frame/PERIOD)+cos(M_PI*((float)x)/_RESOLUTION_X)*cos(2*M_PI*((float)y)/_RESOLUTION_Y*sin(2*SQRT3*M_PI*frame/PERIOD))+0.5*cos(2*M_PI*((float)x)/_RESOLUTION_X)*cos(M_PI*((float)y)/_RESOLUTION_Y)*sin(2*SQRT3*M_PI*frame/PERIOD+1.0f)+cos(M_PI*((float)x)/_RESOLUTION_X)*cos(M_PI*((float)y)/_RESOLUTION_Y)*sin(2*M_PI*frame/PERIOD)+0.1*cos(4*M_PI*((float)x)/_RESOLUTION_X)*cos(4*M_PI*((float)y)/_RESOLUTION_Y)*sin(2*4*SQRT2*M_PI*frame/PERIOD+0.5)+0.1*cos(4*M_PI*((float)x)/_RESOLUTION_X)*cos(2*M_PI*((float)y)/_RESOLUTION_Y)*sin(2*2*SQRT3*M_PI*frame/PERIOD+1.5)+0.1*cos(2*M_PI*((float)x)/_RESOLUTION_X)*cos(4*M_PI*((float)y)/_RESOLUTION_Y)*sin(2*2*SQRT3*M_PI*frame/PERIOD+0.5);
}
int main(){
        VideoWriter outputVideo;
        outputVideo.open("output.mp4" , CV_FOURCC('F', 'M', 'P', '4'), 30,Size(_RESOLUTION_X,_RESOLUTION_Y), true);
        for(int frame=0;frame<600;frame++){
                printf("Frame %d\n",frame);
                render(swave, &frame,outputVideo);
        }
        return 0;
}
