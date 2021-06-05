function [Rnew,tnew] = IRLS_FiniteDiff(R_init, t_init, A, M, m, N, threshold_irls)
% Finite Differences IRLS
% Uses Finite Differences for Jacobian computation
% Robust estimation of camera pose with respect to outliers

R = R_init;
t = t_init;
lamda = 0.001;
u = threshold_irls + 1;

% delta for finite differences
eps = 1e-2;

camera_param = cameraParameters('IntrinsicMat', A);
c = 4.685;

iter = 0;
while ((iter < N) && (u > threshold_irls))
    %%%%%%%%%%%%%% Computation of 'e' %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Rvec = rotationMatrixToVector(R);
    pose = [Rvec t];
    Mproj2d = project3d2image(M, camera_param, R, t);
    diff2d = (Mproj2d-m);
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
    %%%%%%%%%%%%%% Jacobian Computation %%%%%%%%%%%%%%
    dh = zeros(6,1);
    Jacobian = zeros(length(esigma),6);
    for p = 1:6
        dh(p,1) = eps;
        pose_idh = (pose' + dh)';
        pose_ddh = (pose' - dh)';
        fposei = project3d2image(M, camera_param, rotationVectorToMatrix(pose_idh(1:3)), pose_idh(4:6));
        for i = 1:length(fposei)
            if i == 1
                einc = fposei(:,i);
            else
                einc = vertcat(einc, fposei(:,i));
            end
        end
        fposed = project3d2image(M, camera_param, rotationVectorToMatrix(pose_ddh(1:3)), pose_ddh(4:6));
        for i = 1:length(fposed)
            if i == 1
                edec = fposed(:,i);
            else
                edec = vertcat(edec, fposed(:,i));
            end
        end
        dh(p,1) = 0;
        Jacobian(:,p) =  (einc -edec)/(2*eps);
    end
    
    %%%%%%%%%%%%% Computation of 'delta' %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    delta = -((inv((Jacobian' * W * Jacobian) + (lamda * eye(6)))) * (Jacobian' * W * e));
    
    %%%%%%%%%%%% Compute update %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    pose_new = pose' + delta;
    Rnew = rotationVectorToMatrix(pose_new(1:3));
    tnew = pose_new(4:6);
    Mproj2dnew = project3d2image(M, camera_param, Rnew, tnew');
    %Mproj2dnew = worldToImage(camera_param, Rnew, tnew, M);
    diff2dnew = (Mproj2dnew-m);
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
        pose = pose_new';
    end
    u = norm(delta, 2);
    iter = iter + 1;
    
  end
end

