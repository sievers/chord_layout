import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata


def make_uv_coverage(x,y,du=1.0,pad=8):
    u1,u2=np.meshgrid(x,x)
    uu=np.ravel(u1-u2)
    
    v1,v2=np.meshgrid(y,y)
    vv=np.ravel(v1-v2)
    print('U shape is ',uu.shape)

    umax=np.max(uu)
    vmax=np.max(vv)
    umax=(np.ceil((umax+pad)/du)*du)
    vmax=(np.ceil((vmax+pad)/du)*du)

    #uvec=np.linspace(-umax,umax,nu)
    #vvec=np.linspace(-vmax,vmax,nv)
    nu=int(np.round(1+2*umax/du))
    nv=int(np.round(1+2*vmax/du))

    grid=np.zeros([nu,nv])
    uvec=np.asarray(np.round(uu/du),dtype='int')
    vvec=np.asarray(np.round(vv/du),dtype='int')
    for i in range(len(uvec)):
        grid[uvec[i],vvec[i]]=grid[uvec[i],vvec[i]]+1
    return grid

def remove_plan(x,y,z):
    mat=np.empty([len(x),3])
    mat[:,0]=1
    mat[:,1]=x
    mat[:,2]=y
    lhs=mat.T@mat
    rhs=mat.T@z
    fitp=np.linalg.inv(lhs)@rhs
    resid=z-mat@fitp
    return resid
def lineseg_dists(p, a, b):
    """Cartesian distance from point to line segment

    Edited to support arguments as series, from:
    https://stackoverflow.com/a/54442561/11208892

    Args:
        - p: np.array of single point, shape (2,) or 2D array, shape (x, 2)
        - a: np.array of shape (x, 2)
        - b: np.array of shape (x, 2)
    """
    # normalized tangent vectors
    d_ba = b - a
    d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1])
                           .reshape(-1, 1)))

    # signed parallel distance components
    # rowwise dot products of 2D vectors
    s = np.multiply(a - p, d).sum(axis=1)
    t = np.multiply(p - b, d).sum(axis=1)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros(len(s))])

    # perpendicular distance component
    # rowwise cross products of 2D vectors  
    d_pa = p - a
    c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]

    return np.hypot(h, c)

def point_seg_dist(x,y,x1,y1,x2,y2):
    dx1=x1-x
    dy1=y1-y
    dx2=x2-x
    dy2=y2-y

    d1=(dx1**2+dy1**2)**0.5
    d2=(dx2**2+dy2**2)**0.5


    dx=x2-x1
    dy=y2-y1
    mycross=dx2*dy-dy2*dx
    d3=mycross/(dx**2+dy**2)**0.5
    print('dists are ',d1,d2,d3)

    if False:
        m=(y2-y1)/(x2-x1)
        b=y1-m*x1
        
        M=-1/m
        B=y-M*x
    #now we have y=mx+b and y=Mx+B for the perpendicular.  
    #intersection is 0=(m-M)x+(b-B)
        xx=(B-b)/(m-M)
        yy=m*xx+b
        d3=( (xx-x)**2+(yy-y)**2)**0.5
        
        if ((xx>x1)&(xx>x2))|((xx<x1)&(xx<x2)):
            dists=np.asarray([d1,d2])
        else:
            dists=np.asarray([d1,d2,d3])
    return np.min([d1,d2,d3])
    
def is_interior(x,y,points,wrap=True):
    #check if a point is in the interior of the polygon formed by tracing out points
    #we'll do this by extending a ray in the +y direction and seeing if it intersects an even
    #or odd number of edges
    ncross=0
    npoints=points.shape[0]
    if wrap:
        imin=1
    else:
        imin=0
    for i in range(imin,npoints):
        #if we are to the left of both x points, or to the right of both 
        #x points, our vertical line will no hit the line segment
        if (x<points[i,0])&(x<points[i-1,0]):
            continue
        if (x>points[i,0])&(x>points[i-1,0]):
            continue
        dy=points[i,1]-points[i-1,1]
        dx=points[i,0]-points[i-1,0]
        if np.abs(dy)<np.abs(dx): 
            m=dy/dx
            b=points[i,1]-m*points[i,0]
            #we have y=mx+b, find the y value for the input x
            y_int=m*x+b
            if y_int>y:
                ncross=ncross+1
        else:
            m=dx/dy
            b=points[i,0]-m*points[i,1]
            #we have x=my+b
            y_int=(x-b)/m
            if y_int>y:
                ncross=ncross+1
    return (ncross%2)==1  #if we had an odd number of intersections, we're inside
        
plt.ion()
#points=np.loadtxt('chord_envelope.txt')
#points=np.loadtxt('envelope_between_fences.txt')
#points=np.loadtxt('CHORD_envelope_revised.csv',skiprows=1,delimiter=',')
#points=np.loadtxt('CHORD_envelope_revised_2.txt')
#points=points[:,2:]
points=np.loadtxt('revised max envelope.csv',delimiter=',',skiprows=1)

mymean=np.mean(points,axis=0)
points[:,0]=points[:,0]-mymean[0]
points[:,1]=points[:,1]-mymean[1]

plt.clf();
plt.plot(points[:,0],points[:,1],'b')
plt.plot(points[:,0],points[:,1],'b*')
plt.show()


dx=6.3
dy=8.5
x=points[:,0]
y=points[:,1]
xvec=np.arange(x.min(),x.max(),dx)-1.5
yvec=np.arange(y.min(),y.max(),dy)

xgap=0.5*dx*1
ygap=0.5*dy*1
xthresh=np.median(xvec) #was 20
ythresh=np.median(yvec) #was -5
xvec[xvec>xthresh]=xvec[xvec>xthresh]+xgap
yvec[yvec<ythresh]=yvec[yvec<ythresh]-ygap
#yvec[yvec>-5]=yvec[yvec>-5]+ygap


theta=1.8*np.pi/180
rotmat=np.empty([2,2])
rotmat[0,0]=np.cos(theta)
rotmat[1,1]=np.cos(theta)
rotmat[0,1]=-np.sin(theta)
rotmat[1,0]=np.sin(theta)

yy,xx=np.meshgrid(yvec,xvec)
yy=np.reshape(yy,yy.size)
xx=np.reshape(xx,xx.size)
xy=np.zeros([len(xx),2])
xy[:,0]=xx
xy[:,1]=yy
xy=xy@rotmat
#xvec=x[:,0]
#yvec=y[:,1]
xx=xy[:,0]
yy=xy[:,1]


isinside=np.zeros(xx.shape,dtype='bool')
if False:
    for i in range(len(xvec)):
        for j in range(len(yvec)):
            isinside[i,j]=is_interior(xvec[i],yvec[j],points)
else:
    for i in range(len(xx)):
        isinside[i]=is_interior(xx[i],yy[i],points)



xx=xx[isinside]
yy=yy[isinside]
dist=np.zeros(len(xx))
xxyy=np.vstack([xx,yy]).T
#dists=lineseg_dists(xxyy,points[:-1,:],points[1:,:])
for i in range(len(xx)):
    tmp=lineseg_dists(xxyy[i,:],points[:-1,:],points[1:,:])
    dist[i]=tmp.min()
#mask=dist>5.1
mask=dist>3.2
xx=xx[mask]
yy=yy[mask]
plt.plot(xx,yy,'r.')


#crud=np.loadtxt('chord_beween_fences.txt')
crud=np.loadtxt('chord_site_survey_bigarea.txt')
#
crud[:,1]=crud[:,1]-mymean[1]
crud[:,2]=crud[:,2]-mymean[0]
crud[:,0]=crud[:,3]-np.mean(crud[:,3])

mat=np.zeros([crud.shape[0],3])
mat[:,0]=1
mat[:,1]=crud[:,1]
mat[:,2]=crud[:,2]
lhs=mat.T@mat
rhs=mat.T@crud[:,3]
fitp=np.linalg.inv(lhs)@rhs
resid=crud[:,3]-mat@fitp

thresh1=-0.35 #-0.35
thresh2=-0.3  #-0.3

mask=resid>thresh1 #default was -0.35

#redo the plane removal now that we have a better guess
#which points we want to cut
lhs=mat[mask,:].T@mat[mask,:]
rhs=mat[mask,:].T@crud[mask,3]
fitp=np.linalg.inv(lhs)@rhs
resid=crud[:,3]-mat@fitp



tmp=np.vstack([crud[:,2],crud[:,1]]).T
tmp2=np.vstack([xx,yy]).T

zz=griddata(tmp,resid,tmp2)

mask=(np.isfinite(zz))&(zz>thresh1)


xx_use=xx[mask]
yy_use=yy[mask]
zz_use=zz[mask]
zz_use=remove_plan(xx_use,yy_use,zz_use)

mask=zz_use>thresh2
xx_use=xx_use[mask]
yy_use=yy_use[mask]
zz_use=zz_use[mask]

#plt.scatter(xx[mask],yy[mask],c=zz[mask],s=60)
plt.scatter(xx_use,yy_use,c=zz_use,s=60)

plt.savefig('chord_layout_polygon_fences.png')

xtmp=xx_use+mymean[0]
ytmp=yy_use+mymean[1]
grid=make_uv_coverage(xtmp,ytmp,0.5)
f=open('chord_dish_coords_fences_wgap_clean.txt','w')
for i in range(len(xtmp)):
    f.write(repr(xtmp[i])+' '+repr(ytmp[i])+'\n')
f.close()

xy_tmp=np.vstack([xx_use,yy_use]).T
xy_tmp_unrot=xy_tmp@np.linalg.inv(rotmat)

xmax=np.max(xy_tmp_unrot[:,0])
mask=xy_tmp_unrot[:,0]<xmax-0.1
xy_tmp_unrot=xy_tmp_unrot[mask,:]

#ymax=np.max(xy_tmp_unrot[:,0])
#mask=xy_tmp_unrot[:,0]<ymax-0.1
#xy_tmp_unrot=xy_tmp_unrot[mask,:]
xy_tmp=xy_tmp_unrot@rotmat
print('have ',xy_tmp.shape[0],' dishes now')
xtmp2=xy_tmp[:,0]+mymean[0]
ytmp2=xy_tmp[:,1]+mymean[1]
plt.plot(xy_tmp[:,0],xy_tmp[:,1],'k*')
plt.title(repr(len(xtmp2))+' Good Dishes, Fence Moved')
plt.savefig('chord_moved_fences_trimmed.png')

f=open('chord_dish_coords_fences_wgap_clean_trimmed.txt','w')
for i in range(len(xtmp2)):
    f.write(repr(xtmp2[i])+' '+repr(ytmp2[i])+'\n')
f.close()



xtmp=xx+mymean[0]
ytmp=yy+mymean[1]
f=open('chord_dish_coords_fences_wgap_all.txt','w')
for i in range(len(xtmp)):
    f.write(repr(xtmp[i])+' '+repr(ytmp[i])+'\n')
f.close()

