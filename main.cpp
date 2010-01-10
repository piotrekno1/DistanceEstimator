/*
 * Piotr Duda, Jakub Matraszek
 * Internetowe Systemy Pomiarowe projekt zaliczeniowy. Estymacja odleglosci
 * glowy od kamery rejestrujacej.
 *
 * uzycie : detector --haar nazwa_pliku1 --out-file nazwa_pliku2 [--no-gui --fps
 * num]
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <time.h>
#include <ctype.h>

#include "cv.h"
#include "highgui.h"


/* GUI window name and title */
#define WINDOW_NAME     "Face distance estimator"

/* Number of the camera device. -1 enables openCV to use
 * any available device */
#define CAM_DEV_NUM     -1          

/*  Close the application when this kbd scan code is pressed */
#define CLOSE_BUTTON    27

/* Minimal frame delay time */
#define MIN_FRAME_DELAY 3

/* cvHaarDetectObjects parameters, described in detail
 * in the openCV documentation */
#define SCALE_FACTOR    1.1
#define MIN_NEIGHBOURS  2 
#define FACE_MIN_SIZE   55

/*
 * A buffer length for converting a range to string
 */
#define NUM_FACES_BUF  13 

/*  Marging for displaying the face number */
#define FACE_NUM_MARGIN 5

/* Margin for displaying face distances */
#define DISTANCE_LIST_MARGIN 10

/* vertical distance between rows */
#define ROW_VERTICAL 15

/*  horizontal distance between colums */
#define COL_HORIZONTAL 50

/* Maximal number of faces processed by the algorithm */
#define MAX_FACES   10

#define DEFAULT_FPS 10

/* Number of size:disance measurments */
#define NUM_MEASURMENTS 22


struct distance_range
{
    int min; /*  minimal object distance from the camera */
    int max; /*  maximal object distance  */
};


int distance_cm[NUM_MEASURMENTS] = {
     30, 40, 50, 60, 70, 80, 90,
    100,110,120,130,140,150,160,
    170,180,190,200,230,260,290,
    400
};

/*
 * Those values depend on the camera's resolution itd. If you are not
 * using A4-Tech pk835 please measure them with your camera.
 * To do so you need to check the width of the face selection window for 
 * each distance, found in the distance_cm[].
 */
int size_pixel[NUM_MEASURMENTS]={
    403,330,278,241,210,195,177,
    158,144,132,121,115,103, 99,
     96, 90, 86, 82, 75, 63, 56,
     51
};


/* Memory pool for calculations */
static CvMemStorage *storage = 0;

/* Memory for a Haar classifer for face detection */
static CvHaarClassifierCascade *cascade = 0;

/* Font for on screen display */
static CvFont font;

/*  Path to the cascade data file */
char *haar_data_file = NULL;

/* Output file - prints out faces and distances */
char *output_file = NULL;

/* Should the application work in the GUI mode? */
bool no_gui = false;

/* Number of frames per second handled by the app */
int actual_fps = 0;

/* Number of frames per second requested by the user */
int wanted_fps = DEFAULT_FPS;

void check_cli(int argc,char *argv[]);
void usage(char *argv_0);
void write_to_file(FILE* out_fp, distance_range *distances,int n);

CvSeq* detect_faces(IplImage *img);
int get_distances(CvSeq* faces, distance_range *distances);
distance_range calculate_distance(int width, int height );

void select_faces(IplImage *img, CvSeq* faces);
void draw_distances(IplImage *img, CvSeq* faces);
void draw_fps(IplImage *img,int fps);


int main(int argc, char **argv)
{
    CvCapture *capture  = 0;
    IplImage  *frame    = 0;
    CvSeq     *faces    = 0;
    FILE      *out_fp   = 0;
    
    char  key_pressed = 0;
    int   prev_faces  = 0;
    int   delay_mili  = 0;
    int n;

    clock_t start,end; /*  for measuring fps */

    distance_range distances[MAX_FACES];
    

    check_cli(argc,argv);

    printf("Registering camera input...\n");
    if( ( capture = cvCreateCameraCapture( CAM_DEV_NUM ) ) == NULL )
    {
        printf("Can't connect to the camera!\n");
        exit(1);
    }

    printf("Loading face classifier data...\n");
    cascade = ( CvHaarClassifierCascade *) cvLoad( haar_data_file, 0, 0, 0 );
    if( cascade == NULL )
    {
        printf("Can't load classifier data!\n");
        exit(1);
    }

    printf("Initializating font...\n");
    cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, 1.0, 1.0, 0.0 );

    printf("Opening output file...\n");
    if( (out_fp = fopen(output_file,"w")) == NULL )
    {
        printf("Can't open output file: %s\n",output_file );
        usage(argv[0]);
        return -1;
    }


    /* Allocate calculation memory pool */
    storage = cvCreateMemStorage(0);

    /*  Create display window */
    if(!no_gui)
        cvNamedWindow( WINDOW_NAME, CV_WINDOW_AUTOSIZE );
    
    while(1) 
    {
        start = clock(); 

        frame = cvQueryFrame( capture );
        if(!frame)
            break;

        /* Get all the faces on the current frame */
        faces = detect_faces(frame);

        /* 
         * Differren number of faces detected, make sure it's not a 
         * single frame mistake (they happen randomly). 
         */
        if( faces->total != prev_faces && faces->total > 0)
        {
            /*  reopen file to clear it */
            fclose(out_fp);
            out_fp  = fopen(output_file, "w");
            prev_faces = faces->total;
            continue;
        }

        /*
         * Get distances between faces and the camera
         */
        n = get_distances(faces, distances);
        write_to_file(out_fp, distances, n);

        /*
         * Draw the result on the screen
         */
        if(!no_gui)
        {
            select_faces(frame, faces);
            draw_distances(frame, faces);
            draw_fps(frame,actual_fps);
            cvShowImage(WINDOW_NAME, frame);
        }

        /* 
         * Count the delay time to get wanted number of FPS. Time of execution
         * of previous functions has to be considered.
         */
        end = clock();
        delay_mili = (1/(double)wanted_fps)*1000; 
        delay_mili-= (int)(end-start)/(double)CLOCKS_PER_SEC*1000;
        if(delay_mili < MIN_FRAME_DELAY )
            delay_mili = MIN_FRAME_DELAY;
        
        /*
         * Check if the close button was pressed
         */
        key_pressed = cvWaitKey( delay_mili ) ;
        if( key_pressed == CLOSE_BUTTON) 
            break;
        
        end = clock();
        actual_fps = (int)round(1/((double)(end-start)/(double)CLOCKS_PER_SEC)); 
    }
    
    cvReleaseCapture( & capture );
    
    if(!no_gui)
        cvDestroyWindow( WINDOW_NAME );

    fclose(out_fp);

    return 0;
}


/* 
 * Check for command line parameters 
 */
void check_cli(int argc,char *argv[])
{
    if(argc > 1)
    {
        for( int i=0; i<argc; i++)
        {
            if(!strcmp(argv[i], "--no-gui"))
            {
                printf("Application running in non GUI mode.\n");
                no_gui = true;
            }
            else if(!strcmp(argv[i], "--haar"))
            {
                printf("Used HAAR clasifier: %s\n", argv[i+1]);
                haar_data_file = argv[++i];
            }
            else if(!strcmp(argv[i], "--out-file"))
            {
                printf("Setting ouput file to: %s\n", argv[i+1]);
                output_file = argv[++i];
            }
            else if(!strcmp(argv[i], "--fps"))
            {
                printf("Trying %s fps.\n", argv[i+1]);
                wanted_fps = atoi(argv[++i]);
                if(wanted_fps == 0)
                {
                    printf("Bad fps value. Setting to default.\n");
                    wanted_fps = DEFAULT_FPS;
                }
            }
        }
    }

    if(haar_data_file == NULL || output_file == NULL)
    {
        usage(argv[0]);
        exit(-1);
    }
}


/*
 * Shows the list of input parameters
 */
void usage(char *argv_0)
{
    printf("\nUsage:\n");
    printf("%s --haar filename1 --out-file filename2 [--no-gui --fps num]\n\n",
            argv_0);
}

/*
 * Write the distances to the beginning of the output file
 */
void write_to_file(FILE* out_fp, distance_range *distances,int n)
{
    int i=0;

    fseek(out_fp, 0, SEEK_SET);
    for(; i<n; i++)
        fprintf(out_fp, "%d\t%d-%d\n",i,distances[i].min,distances[i].max);
}


/*
 * Process the input image - find all the faces and reutrn their
 * description.
 * Returns all the detected faces in the form of a CvSeq.
 */
CvSeq* detect_faces(IplImage *img)
{
    CvSeq *faces = NULL;

    /*  Clear storage from previous calculations */
    cvClearMemStorage( storage );

    /*  Check if face data is loaded */
    if( cascade )
    {
        faces = cvHaarDetectObjects( img, cascade, storage,
                SCALE_FACTOR, MIN_NEIGHBOURS, CV_HAAR_DO_CANNY_PRUNING,
                cvSize(FACE_MIN_SIZE, FACE_MIN_SIZE) );
    }

    return faces;
}

int get_distances(CvSeq* faces, distance_range *distances)
{
    int i;
    int end;

    end =  faces ? faces->total : 0 ;
    end =  end > MAX_FACES ? MAX_FACES : end ;

    /* For every face draw */
    for( i = 0; i < end; i++ )
    {
        CvRect *r = (CvRect*)cvGetSeqElem( faces, i );
        distances[i] = calculate_distance( r->width, r->height );
    }
    
    return end;
}


/*
 * Given width and height of the window sorrouding
 * a face, estimate its distance from the camera
 */
distance_range calculate_distance( int width, int height )
{
    int i;
    int margin;
    distance_range range;
    
    /* if window size is larger than maximal in the table */
    if( width >= size_pixel[0])
    {
        range.min = 0;
        range.max = distance_cm[0];
    }
   
    /* if window size if smaller than minimal in the table */
    else if( width <= size_pixel[NUM_MEASURMENTS-1])
    {
        range.min = range.max = distance_cm[NUM_MEASURMENTS-1];
    }

    else{
        for(i = 1; i< NUM_MEASURMENTS-1; i++)
        {
            if(width <= size_pixel[i] && width >= size_pixel[i+1])
            {
                margin = (distance_cm[i+1] - distance_cm[i])/2;
                range.min = distance_cm[i] - margin;
                range.max = distance_cm[i+1] + margin;
                break;
            }
        }
    }

    return range;
}

/*
 * Sorrounds detected faces with a green rectangle.
 */
void select_faces(IplImage *img, CvSeq* faces)
{
    /*  Points for drawing rectangle */
    CvPoint pt1,pt2;
    int i;

    /* For every face draw */
    for( i = 0; i < ( faces ? faces->total : 0 ); i++ )
    {
        /*  A rectangle border - sorrounding the face  */
        CvRect *r = (CvRect*)cvGetSeqElem( faces, i );
        pt1.x = r->x;
        pt2.x = r->x + r->width;
        pt1.y = r->y;
        pt2.y = r->y+r->height;

        cvRectangle( img, pt1, pt2, CV_RGB(0,255,0), 3, 8, 0 );
    }
}

/*
 * Draws the distances between face and the camera. Uses calculate_distance.
 */
void draw_distances(IplImage *img, CvSeq* faces)
{
    /* Point for distance list */
    CvPoint pt1;
    distance_range range;

    int i;
    char buf[NUM_FACES_BUF]; 

    memset( buf, 0, sizeof(buf) );

    /* Distance for the face number */
    pt1.x = DISTANCE_LIST_MARGIN;
    pt1.y = 2*DISTANCE_LIST_MARGIN;
    cvPutText( img, "Face number and its distance:", pt1, &font, CV_RGB(0,0,255) ); 

    /* For every face draw */
    for( i = 0; i < ( faces ? faces->total : 0 ); i++ )
    {
        CvRect *r = (CvRect*)cvGetSeqElem( faces, i );
        
        /*  Face number */
        sprintf(buf,"%d",i);

        pt1.y += ROW_VERTICAL;
        cvPutText( img, buf, pt1, &font, CV_RGB(255, 0, 0) );
       
        range = calculate_distance(r->width, r->height);
        sprintf(buf,"~(%d : %d)", range.min, range.max);
        pt1.x += COL_HORIZONTAL;
        cvPutText( img, buf, pt1, &font, CV_RGB(255, 0, 0 ) );
        pt1.x -= COL_HORIZONTAL;

    }
}

void draw_fps(IplImage *img,int fps)
{
    CvPoint pt;
    char buf[NUM_FACES_BUF];
    sprintf(buf,"FPS: %d",fps);
    pt.x = img->width-70;
    pt.y = ROW_VERTICAL;
    cvPutText( img, buf, pt, &font, CV_RGB(255, 0, 0) );    
}

