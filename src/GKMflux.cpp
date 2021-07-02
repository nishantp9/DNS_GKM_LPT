/*****************************************************************************************************************
 * Flux evaluation using Gas-Kinetic scheme
 * --------------------------------------------------------------------------------------------------------------*
 *****************************************************************************************************************/
#include "GKMflux.h"

using namespace std;
/*****************************************************************************************************************
 * Main flux function	
 *****************************************************************************************************************/
void flux(ptype WL[5], ptype WR[5], ptype Wl[5], ptype Wr[5], ptype DWxl[5], ptype DWxr[5], 
		  ptype DWyl[5], ptype DWyr[5], ptype DWzl[5], ptype DWzr[5], ptype tau, ptype dt, ptype dx, ptype F[5])
{
/* ----------------------------------------------------------------------------------------------------------------------------------*
 * Variables Decleration
 * ----------------------------------------------------------------------------------------------------------------------------------*/	
	ptype Pl, denl, laml, Ul[3], Pr, denr, lamr, Ur[3], We[5], Pe, dene, lame, Ue[3], PL, denL, lamL, UL[3], PR, denR, lamR, UR[3];	
	ptype Ie2l, Ie4l, Ie2r, Ie4r, Ipl[3][7], Ifl[3][7], Inr[3][7], Ifr[3][7], Ie2e, Ie4e, Ipe[3][7], Ine[3][7], Ife[3][7];

	ptype Mpgl000[5], Mngr000[5];
	ptype Mfgl100_axl[5], Mfgr100_axr[5], Mfgl010_ayl[5], Mfgr010_ayr[5], Mfgl001_azl[5], Mfgr001_azr[5];

	ptype bxl[5], bxr[5], byl[5], byr[5], bzl[5], bzr[5], bx_l[5], bx_r[5], by_[5], bz_[5], Bl[5], Br[5], B_[5];
	ptype axl[5], axr[5], ayl[5], ayr[5], azl[5], azr[5], ax_l[5], ax_r[5], ay_[5], az_[5], Al[5], Ar[5], A_[5];

	ptype Mpgl000_ayl[5], Mngr000_ayr[5], Mpgl000_azl[5], Mngr000_azr[5];
	ptype w, gm0, gm1, gm2, gm3, gm4, gm5;

	ptype Mfge000[5], Mpge100_ax_l[5], Mnge100_ax_r[5], Mfge010_ay_[5], Mfge001_az_[5], Mpgl100_axl[5], 
          Mngr100_axr[5], Mpgl010_ayl[5], Mngr010_ayr[5], Mpgl001_azl[5], Mngr001_azr[5], Mpgl000_Al[5], Mngr000_Ar[5];

	ptype P, Q, p0, p1, p2, p3, p4, p5;

	ptype Mfge100[5], Mpgl100[5], Mngr100[5], Mpge200_ax_l[5], Mfge100_A_[5], Mnge200_ax_r[5], Mfge110_ay_[5], Mfge101_az_[5], 
	      Mpgl200_axl[5], Mngr200_axr[5], Mpgl110_ayl[5], Mngr110_ayr[5], Mpgl101_azl[5], Mngr101_azr[5], Mpgl100_Al[5], Mngr100_Ar[5];			

/* ------------------------------------------------------------------*
 * INTEGRATIONS (MOMENT CALCULATIONS) [l & r]
 * ------------------------------------------------------------------*/
	c2p(Wl, denl, Ul, Pl);	c2p(Wr, denr, Ur, Pr);
	laml  = 0.5*denl/Pl;	lamr  = 0.5*denr/Pr;
	
	Ie2l = K/(2*laml);
	Ie4l = 3*K/(4*laml*laml) + K*(K-1)/(4*laml*laml);
	Ie2r = K/(2*lamr);
	Ie4r = 3*K/(4*lamr*lamr) + K*(K-1)/(4*lamr*lamr);
	
	for(int j=0; j<3; j++)
	{
		Ifl[j][0] = 1.0;
		Ifl[j][1] = Ul[j];
		Ifr[j][0] = 1.0;
		Ifr[j][1] = Ur[j];
		Ipl[j][0] = 0.5*(erfc(-sqrt(laml)*Ul[j]));
		Inr[j][0] = 0.5*(erfc(sqrt(lamr)*Ur[j]));
		Ipl[j][1] = Ul[j]*Ipl[j][0] + 0.5*exp(-laml*Ul[j]*Ul[j])/sqrt(laml*pi);
		Inr[j][1] = Ur[j]*Inr[j][0] - 0.5*exp(-lamr*Ur[j]*Ur[j])/sqrt(lamr*pi);

		for(int i=2; i<7; i++)
		{
			Ipl[j][i] = Ul[j]*Ipl[j][i-1] + Ipl[j][i-2]*(i-1)/(2*laml);
			Ifl[j][i] = Ul[j]*Ifl[j][i-1] + Ifl[j][i-2]*(i-1)/(2*laml);
			Inr[j][i] = Ur[j]*Inr[j][i-1] + Inr[j][i-2]*(i-1)/(2*lamr);
			Ifr[j][i] = Ur[j]*Ifr[j][i-1] + Ifr[j][i-2]*(i-1)/(2*lamr);
		}
	}

/* ------------------------------------------------------------------*
 * W0 CALCULATION
 * ------------------------------------------------------------------*/
	MCal(Mpgl000, Ipl, Ifl, Ie2l, Ie4l, 0, 0, 0);
	MCal(Mngr000, Inr, Ifr, Ie2r, Ie4r, 0, 0, 0);
	
	FOR(q, 5)
		We[q] = denl * Mpgl000[q] + denr * Mngr000[q];
	
	c2p(We, dene, Ue, Pe);	lame  = 0.5*dene/Pe;

/* ------------------------------------------------------------------*
 * INTEGRATIONS (MOMENT CALCULATIONS) [e]
 * ------------------------------------------------------------------*/
	Ie2e = K/(2*lame);
	Ie4e = 3*K/(4*lame*lame) + K*(K-1)/(4*lame*lame);
	
	for(int j=0; j<3; j++) {
		Ife[j][0] = 1.0;
		Ife[j][1] = Ue[j];
		Ipe[j][0] = 0.5*(erfc(-sqrt(lame)*Ue[j]));
		Ine[j][0] = 0.5*(erfc(sqrt(lame)*Ue[j]));
		Ipe[j][1] = Ue[j]*Ipe[j][0] + 0.5*exp(-lame*Ue[j]*Ue[j])/sqrt(lame*pi);
		Ine[j][1] = Ue[j]*Ine[j][0] - 0.5*exp(-lame*Ue[j]*Ue[j])/sqrt(lame*pi);

		for(int i=2; i<7; i++) {
			Ipe[j][i] = Ue[j]*Ipe[j][i-1] + Ipe[j][i-2]*(i-1)/(2*lame);
			Ife[j][i] = Ue[j]*Ife[j][i-1] + Ife[j][i-2]*(i-1)/(2*lame);
			Ine[j][i] = Ue[j]*Ine[j][i-1] + Ine[j][i-2]*(i-1)/(2*lame);
		}
	}
/* --------------------------------------------------------------------------*
 * Slope Calculation [phase I]: (axl, axr, ayl, ayr, azl, azr, ax_l, ax_r)
 * --------------------------------------------------------------------------*/
	FOR(i, 5) {
		bxl[i] = DWxl[i]/denl;
		bxr[i] = DWxr[i]/denr;		
		byl[i] = DWyl[i]/denl;
		byr[i] = DWyr[i]/denr;		
		bzl[i] = DWzl[i]/denl;		
		bzr[i] = DWzr[i]/denr;		
		bx_l[i] = 2*(We[i] - WL[i]) / (dene*dx);
		bx_r[i] = 2*(WR[i] - We[i]) / (dene*dx);
	}
	slopesolver(bxl, Ul, laml, axl);
	slopesolver(bxr, Ur, lamr, axr);
	slopesolver(byl, Ul, laml, ayl);
	slopesolver(byr, Ur, lamr, ayr);
	slopesolver(bzl, Ul, laml, azl);
	slopesolver(bzr, Ur, lamr, azr);
	slopesolver(bx_l, Ue, lame, ax_l);
	slopesolver(bx_r, Ue, lame, ax_r);

/* -------------------------------------------------------------------------*
 * Slope Calculation [phase II]: (ay_, az_, Al, Ar)
 * -------------------------------------------------------------------------*/
	MCal(Mpgl000_ayl, Ipl, Ifl, Ie2l, Ie4l, 0, 0, 0, ayl);
	MCal(Mngr000_ayr, Inr, Ifr, Ie2r, Ie4r, 0, 0, 0, ayr);
	MCal(Mpgl000_azl, Ipl, Ifl, Ie2l, Ie4l, 0, 0, 0, azl);
	MCal(Mngr000_azr, Inr, Ifr, Ie2r, Ie4r, 0, 0, 0, azr);
	MCal(Mfgl100_axl, Ifl, Ifl, Ie2l, Ie4l, 1, 0, 0, axl);
	MCal(Mfgr100_axr, Ifr, Ifr, Ie2r, Ie4r, 1, 0, 0, axr);
	MCal(Mfgl010_ayl, Ifl, Ifl, Ie2l, Ie4l, 0, 1, 0, ayl);
	MCal(Mfgr010_ayr, Ifr, Ifr, Ie2r, Ie4r, 0, 1, 0, ayr);
	MCal(Mfgl001_azl, Ifl, Ifl, Ie2l, Ie4l, 0, 0, 1, azl);
	MCal(Mfgr001_azr, Ifr, Ifr, Ie2r, Ie4r, 0, 0, 1, azr);
	
	FOR(i, 5) {
		by_[i] = Mpgl000_ayl[i] + Mngr000_ayr[i];
		bz_[i] = Mpgl000_azl[i] + Mngr000_azr[i];	
		Bl[i] = -Mfgl100_axl[i] - Mfgl010_ayl[i] - Mfgl001_azl[i];		 
		Br[i] = -Mfgr100_axr[i] - Mfgr010_ayr[i] - Mfgr001_azr[i];		 
	}
	slopesolver(by_, Ue, lame, ay_);
	slopesolver(bz_, Ue, lame, az_);
	slopesolver(Bl, Ul, laml, Al);
	slopesolver(Br, Ur, lamr, Ar);
	
/* -------------------------------------------------------------------------*
 * Collision time scale calculation (tau)
 * -------------------------------------------------------------------------*/
	w = abs(denl/laml - denr/lamr)/(abs(denl/laml + denr/lamr));
    tau = tau + dt*w;

/* -------------------------------------------------------------------------*
 * Slope Calculation [phase III]: (A_)
 * -------------------------------------------------------------------------*/
	gm0 = dt - tau*(1-exp(-dt/tau));	gm1 = -(1-exp(-dt/tau))/gm0;
	gm2 = (-dt+2*tau*(1-exp(-dt/tau))-dt*exp(-dt/tau))/gm0;	gm3 = -gm1;
	gm4 = (dt*exp(-dt/tau) -tau*(1-exp(-dt/tau)))/gm0;		gm5 = tau*gm3;
	
	MCal(Mfge000, Ife, Ife, Ie2e, Ie4e, 0, 0, 0);
	MCal(Mpge100_ax_l, Ipe, Ife, Ie2e, Ie4e, 1, 0, 0, ax_l);	
	MCal(Mnge100_ax_r, Ine, Ife, Ie2e, Ie4e, 1, 0, 0, ax_r);	
	MCal(Mfge010_ay_,  Ife, Ife, Ie2e, Ie4e, 0, 1, 0, ay_);	
	MCal(Mfge001_az_,  Ife, Ife, Ie2e, Ie4e, 0, 0, 1, az_);	
	MCal(Mpgl100_axl,  Ipl, Ifl, Ie2l, Ie4l, 1, 0, 0, axl);	
	MCal(Mngr100_axr,  Inr, Ifr, Ie2r, Ie4r, 1, 0, 0, axr);	
	MCal(Mpgl010_ayl,  Ipl, Ifl, Ie2l, Ie4l, 0, 1, 0, ayl);	
	MCal(Mngr010_ayr,  Inr, Ifr, Ie2r, Ie4r, 0, 1, 0, ayr);	
	MCal(Mpgl001_azl,  Ipl, Ifl, Ie2l, Ie4l, 0, 0, 1, azl);	
	MCal(Mngr001_azr,  Inr, Ifr, Ie2r, Ie4r, 0, 0, 1, azr);	
	MCal(Mpgl000_Al,   Ipl, Ifl, Ie2l, Ie4l, 0, 0, 0, Al);	
	MCal(Mngr000_Ar,   Inr, Ifr, Ie2r, Ie4r, 0, 0, 0, Ar);
	
	FOR(i, 5) {
		B_[i] = gm1*dene*Mfge000[i] + gm2*dene*(Mpge100_ax_l[i]+Mnge100_ax_r[i]+Mfge010_ay_[i]+Mfge001_az_[i]) + gm3*(denl*Mpgl000[i]+denr*Mngr000[i]) +
		       (gm4+gm5)*( denl*(Mpgl100_axl[i]+Mpgl010_ayl[i]+Mpgl001_azl[i]) + denr*(Mngr100_axr[i]+Mngr010_ayr[i]+Mngr001_azr[i]) ) +
		        gm5*(denl*Mpgl000_Al[i]+denr*Mngr000_Ar[i]);
		       		
		B_[i] = B_[i] / dene;
	}
	slopesolver(B_, Ue, lame, A_);
	
/* ---------------------------------------------------------------------------------------------------------*
 * Flux calculation	
 * ---------------------------------------------------------------------------------------------------------*/
//	Integral dt
	P  = -tau*(exp(-dt/tau)-1);
	Q  = -tau*dt*exp(-dt/tau)-tau*tau*(exp(-dt/tau)-1);
	p0 = (dt-P);			p1 = 0.5*dt*dt - tau*(p0);
	p2 = -tau*(p0) + Q;		p3 = P;
	p4 = -Q - tau*P;		p5 = -tau*P;
	
	MCal(Mpgl100, Ipl, Ifl, Ie2l, Ie4l, 1, 0, 0);
	MCal(Mngr100, Inr, Ifr, Ie2r, Ie4r, 1, 0, 0);
	MCal(Mfge100, Ife, Ife, Ie2e, Ie4e, 1, 0, 0);
	MCal(Mfge100_A_,   Ife, Ife, Ie2e, Ie4e, 1, 0, 0, A_);	
	MCal(Mpge200_ax_l, Ipe, Ife, Ie2e, Ie4e, 2, 0, 0, ax_l);	
	MCal(Mnge200_ax_r, Ine, Ife, Ie2e, Ie4e, 2, 0, 0, ax_r);	
	MCal(Mfge110_ay_,  Ife, Ife, Ie2e, Ie4e, 1, 1, 0, ay_);	
	MCal(Mfge101_az_,  Ife, Ife, Ie2e, Ie4e, 1, 0, 1, az_);	
	MCal(Mpgl200_axl,  Ipl, Ifl, Ie2l, Ie4l, 2, 0, 0, axl);	
	MCal(Mngr200_axr,  Inr, Ifr, Ie2r, Ie4r, 2, 0, 0, axr);	
	MCal(Mpgl110_ayl,  Ipl, Ifl, Ie2l, Ie4l, 1, 1, 0, ayl);	
	MCal(Mngr110_ayr,  Inr, Ifr, Ie2r, Ie4r, 1, 1, 0, ayr);	
	MCal(Mpgl101_azl,  Ipl, Ifl, Ie2l, Ie4l, 1, 0, 1, azl);	
	MCal(Mngr101_azr,  Inr, Ifr, Ie2r, Ie4r, 1, 0, 1, azr);	
	MCal(Mpgl100_Al,   Ipl, Ifl, Ie2l, Ie4l, 1, 0, 0, Al);	
	MCal(Mngr100_Ar,   Inr, Ifr, Ie2r, Ie4r, 1, 0, 0, Ar);	

	FOR(i, 5) {
		F[i] = p0*dene*Mfge100[i] + p1*dene*Mfge100_A_[i] + p2*dene*(Mpge200_ax_l[i]+Mnge200_ax_r[i]+Mfge110_ay_[i]+Mfge101_az_[i]) + 
		       p3*(denl*Mpgl100[i]+denr*Mngr100[i]) +  p4*( denl*(Mpgl200_axl[i]+Mpgl110_ayl[i]+Mpgl101_azl[i]) + denr*(Mngr200_axr[i]+Mngr110_ayr[i]+Mngr101_azr[i]) ) + 
		       p5*(denl*Mpgl100_Al[i]+denr*Mngr100_Ar[i]);		
	}

}

/******************************************************************************************************************************************************
 * Main flux functions ends here
 ******************************************************************************************************************************************************/
 
 
/* ---------------------------------------------------------------------------------------------------------------------------------------------------*
 *  Moment Matrix Calculator 
 * ---------------------------------------------------------------------------------------------------------------------------------------------------*/
void MCal(ptype M[5], ptype I[3][7], ptype If[3][7], ptype Ie2, ptype Ie4, int k, int l, int m, ptype ax[5])
{
	ptype val0, val1, val2, val3;
	 
	val0 = 0.5 * ( I[0][2+k]*If[1][l]*If[2][m] + I[0][k]*If[1][2+l]*If[2][m] + I[0][k]*If[1][l]*If[2][2+m] + I[0][k]*If[1][l]*If[2][m]*Ie2  );
	
	M[0] = ax[0]*(I[0][k]*If[1][l]*If[2][m]) + ax[1]*(I[0][1+k]*If[1][l]*If[2][m]) + ax[2]*(I[0][k]*If[1][1+l]*If[2][m]) +
	       ax[3]*(I[0][k]*If[1][l]*If[2][1+m]) + ax[4]*val0 ;	

				  
	val1 = 0.5 * ( I[0][3+k]*If[1][l]*If[2][m] + I[0][1+k]*If[1][2+l]*If[2][m] + I[0][1+k]*If[1][l]*If[2][2+m] +
	 			   I[0][1+k]*If[1][l]*If[2][m]*Ie2  );
	
	
	M[1] = ax[0]*(I[0][1+k]*If[1][l]*If[2][m]) + ax[1]*(I[0][2+k]*If[1][l]*If[2][m]) + ax[2]*(I[0][1+k]*If[1][1+l]*If[2][m]) +
		   ax[3]*(I[0][1+k]*If[1][l]*If[2][1+m]) + ax[4]*val1;
				 
 
   val2 = 0.5 * ( I[0][2+k]*If[1][1+l]*If[2][m] + I[0][k]*If[1][3+l]*If[2][m] + I[0][k]*If[1][1+l]*If[2][2+m] +
				  I[0][k]*If[1][1+l]*If[2][m]*Ie2  );
	
	
	M[2] = ax[0]*(I[0][k]*If[1][1+l]*If[2][m])   + ax[1]*(I[0][1+k]*If[1][1+l]*If[2][m]) + ax[2]*(I[0][k]*If[1][2+l]*If[2][m]) +
		   ax[3]*(I[0][k]*If[1][1+l]*If[2][1+m]) + ax[4]*val2;
				 
	
	val3 = 0.5 * ( I[0][2+k]*If[1][l]*If[2][1+m] + I[0][k]*If[1][2+l]*If[2][1+m] + I[0][k]*If[1][l]*If[2][3+m] + 
	               I[0][k]*If[1][l]*If[2][1+m]*Ie2  );
	
	
	M[3] = ax[0]*(I[0][k]*If[1][l]*If[2][1+m]) + ax[1]*(I[0][1+k]*If[1][l]*If[2][1+m]) + ax[2]*(I[0][k]*If[1][1+l]*If[2][1+m]) +
	       ax[3]*(I[0][k]*If[1][l]*If[2][2+m]) + ax[4]*val3;
				 
				 
	M[4] = 0.25*ax[4]* ( I[0][4+k]*If[1][l]*If[2][m] + I[0][k]*If[1][4+l]*If[2][m] + I[0][k]*If[1][l]*If[2][4+m] +
	                     I[0][k]*If[1][l]*If[2][m]*Ie4 + 2*I[0][2+k]*If[1][2+l]*If[2][m] + 2*I[0][2+k]*If[1][l]*If[2][2+m] +
	                   2*I[0][2+k]*If[1][l]*If[2][m]*Ie2 + 2*I[0][k]*If[1][2+l]*If[2][2+m] + 2*I[0][k]*If[1][2+l]*If[2][m]*Ie2 +
	                   2*I[0][k]*If[1][l]*If[2][2+m]*Ie2 ) +
		   ax[0]*val0 + ax[1]*val1 + ax[2]*val2 + ax[3]*val3;
}

void MCal(ptype M[5], ptype I[3][7], ptype If[3][7], ptype Ie2, ptype Ie4, int k, int l, int m)
{
    ptype val0	= 0.5 * ( I[0][2+k]*If[1][l]*If[2][m] + I[0][k]*If[1][2+l]*If[2][m] + I[0][k]*If[1][l]*If[2][2+m] + I[0][k]*If[1][l]*If[2][m]*Ie2  );
		
	M[0] = I[0][k]*If[1][l]*If[2][m];
	M[1] = I[0][1+k]*If[1][l]*If[2][m];
	M[2] = I[0][k]*If[1][1+l]*If[2][m];
	M[3] = I[0][k]*If[1][l]*If[2][1+m];
	M[4] = val0;
}
/* ----------------------------------------------------------------------*
 *  SLOPE 
 * ----------------------------------------------------------------------*/

void slopesolver(ptype b[5], ptype U[3], ptype lam, ptype a[5])
{
	ptype R2, R3, R4, R5;
	
	R2 = b[1] - U[0]*b[0];
	R3 = b[2] - U[1]*b[0];
	R4 = b[3] - U[2]*b[0];
	R5 = 2*b[4] - b[0]*(U[0]*U[0]+U[1]*U[1]+U[2]*U[2]+(K+3)/(2*lam));
	
	a[4] = (1/PRN)*(R5-2*U[0]*R2-2*U[1]*R3-2*U[2]*R4)*(4*lam*lam)/(K+3);
	a[3] = 2*lam*R4 - U[2]*a[4];
	a[2] = 2*lam*R3 - U[1]*a[4];
	a[1] = 2*lam*R2 - U[0]*a[4];
	a[0] = b[0] - a[1]*U[0] - a[2]*U[1] -a[3]*U[2]-.5*a[4]*(U[0]*U[0] +
	       U[1]*U[1] + U[2]*U[2]+(K+3)/(2*lam));

}

/* -------------------------------------------------------------------------*
 * Conservative  --->  Primitive conversion
 * -------------------------------------------------------------------------*/
void c2p(ptype W[5], ptype &den, ptype U[3], ptype &P)
{
	den = W[0];
	U[0]  = W[1]/den;
	U[1]  = W[2]/den;
	U[2]  = W[3]/den;
	P   = (den*(gam-1))*(W[4]/den - 0.5*(U[0]*U[0]+U[1]*U[1]+U[2]*U[2]));
}
/****************************************************************************************
 * ---------------------------------  END  ---------------------------------------------*
 ****************************************************************************************/
