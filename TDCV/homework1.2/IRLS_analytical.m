function [Rnew, tnew] = IRLS_analytical(R_init, t_init, A, M, m, N, threshold_irls)
%IRLS: this function uses the Tukey estimator
%   Robust estimation of camera pose with respect to outliers

R = R_init;
t = t_init;
lamda = 0.001;
u = threshold_irls + 1;

camera_param = cameraParameters('IntrinsicMatrix', A);
c = 4.685;

iter = 1;
while ((iter < N) && (u > threshold_irls))
    %%%%%%%%%%%%%% Computation of 'e' %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Rvec = rotationMatrixToVector(R);
    theta = [Rvec t];
    Mproj2d = project3d2image(M,camera_param, R, t);
    diff2d = (Mproj2d - m);
    for i = 1:length(Mproj2d)
        if i == 1
            e = diff2d(:,i);
        else
            e = vertcat(e, diff2d(:,i));
        end
    end
    stddev = 1.48257968 * median(abs(e));
    
    %%%%%%%%%%%%%% Computation of 'W' %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % e / standard_dev
    esigma = e / stddev;
    
    wi = zeros(length(esigma), 1);
    for eindex=1:length(esigma)
        if abs(esigma(eindex)) < c
            wi(eindex) = (1 - (esigma(eindex) / c)^2)^2;
        end
    end
    W = diag(wi);
    
    %%%%%%%%%%%%% Tukey's Bisquare M estimator 'E(x,theta)' %%%%%%%%%%%%%
    pi = c^2 / 6 *  ones(length(esigma), 1);
    for eindex = 1:length(esigma)
        if abs(esigma(eindex)) <= c
            pi(eindex) = (c^2 / 6) * (1 - (1 - (esigma(eindex) / c)^2)^3);
        end
    end
    Extheta = sum (pi);
    M = M';
    %%%%%%%%%%%% Computation of 'J' %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     
    vskew = [0 -Rvec(3) Rvec(2); Rvec(3) 0 -Rvec(1); -Rvec(2) Rvec(1) 0];
    e_std_basis=[1 0 0;0 1 0;0 0 1];
    tmp = cross(Rvec, (eye(3) - R) * e_std_basis(:, 1));
    cross1 = [0 -tmp(3) tmp(2); tmp(3) 0 -tmp(1); -tmp(2) tmp(1) 0];
    tmp = cross(Rvec, (eye(3) - R) * e_std_basis(:, 2));
    cross2 = [0 -tmp(3) tmp(2); tmp(3) 0 -tmp(1); -tmp(2) tmp(1) 0];
    tmp = cross(Rvec, (eye(3) - R) * e_std_basis(:, 3));
    cross3 = [0 -tmp(3) tmp(2); tmp(3) 0 -tmp(1); -tmp(2) tmp(1) 0];
    dR_dv1 = (((Rvec(1) * vskew) + cross1) * R) / (norm(Rvec, 2))^2;
    dR_dv2 = (((Rvec(2) * vskew) + cross2) * R) / (norm(Rvec, 2))^2;
    dR_dv3 = (((Rvec(3) * vskew) + cross3) * R) / (norm(Rvec, 2))^2;
    Ju = zeros(1, 6);
    Jv = zeros(1, 6);
    
    for j=1:length(M)
        dM_dp = [(dR_dv1 * M(j,:)'), (dR_dv2 * M(j,:)'), (dR_dv3 * M(j,:)'), eye(3)];
        dmt_dM = A';
        tmp1 = [R t'];
        tmp2 = [M(j,:) 1];
        mt = A' * (tmp1 * tmp2');
        U = mt(1);
        V = mt(2);
        K = mt(3);
        dm_dmt = [1/K, 0, -U/(K^2); 0 , 1/K, -V/(K^2)]; 
        
        J_new = dm_dmt * (dmt_dM * dM_dp);
        Ju = J_new(1, :);
        Jv = J_new(2, :);
        J_new = [Ju; Jv];
        if j == 1
            J = J_new;
        else
            J = vertcat(J, J_new);
        end
    end
    M = M';
    %%%%%%%%%%%%% Computation of 'delta' %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    delta = -(inv((J' * W * J) + (lamda * eye(6)))) * (J' * W * e);
    
    %%%%%%%%%%%% Compute update %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    thetanew = theta' + delta;
    Rnew = rotationVectorToMatrix(thetanew(1:3));
    tnew = thetanew(4:6);
    Mproj2dnew = project3d2image(M, camera_param, Rnew, tnew');
    diff2dnew = (Mproj2dnew - m);
    for i = 1:length(Mproj2dnew)
        if i == 1
            enew = diff2dnew(:,i);
        else
            enew = vertcat(enew, diff2dnew(:,i));
        end
    end
    stddevnew = 1.48257968 * median(abs(enew));
    
    % Computation of E(x, theta+delta)
    esigmanew = enew / stddevnew;
    pinew = c^2 / 6 *  ones(length(esigmanew), 1);
    for eindex = 1:length(esigmanew)
        if abs(esigmanew(eindex)) <= c
            pinew(eindex) = (c^2 / 6) * (1 - (1 - (esigmanew(eindex) / c)^2)^3);
        end
    end
    Exthetanew = sum(pinew);
    
    %%%%%%%%%%%%% Comparing the two E(x, theta) %%%%%%%%%%%%%%%%%%%%%%%%%%%
    if Exthetanew > Extheta
        lamda = lamda * 10;
    else
%         fprintf("entered \n");
        lamda = lamda/10;
        R = Rnew;
        t = tnew';
        theta = thetanew';
    end
    u = norm(delta, 2);
    iter = iter + 1;
    
end

