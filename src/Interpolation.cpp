#include "Interpolation.h"

UBspline_3d_d* get_bsplinep (ptype *data) {
	
    Ugrid x_grid, y_grid, z_grid;
    BCtype_d xBC;
    BCtype_d yBC;
    BCtype_d zBC;
    
    y_grid.start = (sta[0]-3)*dx + 0.5*dx;
    y_grid.end = y_grid.start + (nt_segy-1)*dx;
    y_grid.num = nt_segy;
	yBC.lCode = NATURAL;
	yBC.rCode = NATURAL;

    x_grid.start = (sta[1]-3)*dx + 0.5*dx;
    x_grid.end = x_grid.start + (nt_segx-1)*dx;
    x_grid.num = nt_segx;
	xBC.lCode = NATURAL;
	xBC.rCode = NATURAL;

    z_grid.start = (sta[2]-3)*dx + 0.5*dx;
    z_grid.end = z_grid.start + (nt_segz-1)*dx;
    z_grid.num = nt_segz;
	zBC.lCode = NATURAL;
	zBC.rCode = NATURAL;

	UBspline_3d_d *spline_3d = create_UBspline_3d_d(y_grid, x_grid, z_grid, yBC, xBC, zBC, data);

    return spline_3d;
}

UBspline_3d_d* get_bsplined (ptype *data) {
	
    Ugrid x_grid, y_grid, z_grid;
    BCtype_d xBC;
    BCtype_d yBC;
    BCtype_d zBC;
    
    y_grid.start = (sta[0]-1)*dx + 0.5*dx;
    y_grid.end = y_grid.start + (nt_segy-5)*dx;
    y_grid.num = nt_segy-4;
	yBC.lCode = NATURAL;
	yBC.rCode = NATURAL;

    x_grid.start = (sta[1]-1)*dx + 0.5*dx;
    x_grid.end = x_grid.start + (nt_segx-5)*dx;
    x_grid.num = nt_segx-4;
	xBC.lCode = NATURAL;
	xBC.rCode = NATURAL;

    z_grid.start = (sta[2]-1)*dx + 0.5*dx;
    z_grid.end = z_grid.start + (nt_segz-5)*dx;
    z_grid.num = nt_segz-4;
	zBC.lCode = NATURAL;
	zBC.rCode = NATURAL;

	UBspline_3d_d *spline_3d = create_UBspline_3d_d(y_grid, x_grid, z_grid, yBC, xBC, zBC, data);

    return spline_3d;
}
