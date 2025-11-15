%% UR5 IK/FK Demo (Simplified)
clc; clear;

%% Define robot kinematics
zv = [0;0;0];
ex = [1;0;0];
ey = [0;1;0];
ez = [0;0;1];

kin.H = [ez -ey -ey -ey -ez -ey];
kin.P = [0.1625*ez zv -0.425*ex -0.3922*ex -0.1333*ey-0.0997*ez zv -0.0996*ey];
kin.joint_type = zeros([6 1]);
R_6T = rot(ex, deg2rad(90));

%% ====== INPUT: target translation & rotation ======
p_target = [0.3; 0; 0.2];               % End-effector position in meters
R_target = [
    0  1  0;
    1  0  0;
    0  0 -1
];   % End-effector orientation

%% ====== Compute IK ======
[Q_solutions, is_LS_vec] = IK_UR(R_target, p_target, kin);

disp('Number of IK solutions found:'), disp(size(Q_solutions,2));

%% ====== Check FK for each solution ======
for i = 1:size(Q_solutions,2)
    q_sol = Q_solutions(:,i);
    [R_check, p_check] = fwdkin(kin, q_sol);

    disp(['Solution ', num2str(i), ':']);
    disp('Joint angles (rad):'); disp(q_sol');
    disp('Fk backtrace:'); disp(p_check); disp(R_check);
    disp('Position error:'); disp(p_check - p_target);
    disp('Rotation error:'); disp(R_check - R_target);
    disp('------------------------------------');
end

%% ====== IK Function ======
function [Q, is_LS_vec] = IK_UR(R_06, p_0T, kin)
P = kin.P; H = kin.H; Q=[]; is_LS_vec=[];
p_06 = p_0T - P(:,1) - R_06*P(:,7);

[theta1, theta1_is_ls] = sp_4(H(:,2), p_06, -H(:,1), H(:,2)'*sum(P(:,2:5), 2));

for i_t1 = 1:length(theta1)
    q_1 = theta1(i_t1);
    R_01 = rot(H(:,1), q_1);

    [theta5, theta5_is_ls] = sp_4(H(:,2),H(:,6),H(:,5), H(:,2)' * R_01' * R_06 * H(:,6));

    for i_t5 = 1:length(theta5)
        q_5 = theta5(i_t5);
        R_45 = rot(H(:,5), q_5);

        [theta_14, theta_14_is_LS] = sp_1(R_45*H(:,6), R_01'*R_06*H(:,6), H(:,2));
        [q_6, q_6_is_LS] = sp_1(R_45'*H(:,2), R_06'*R_01*H(:,2), -H(:,6));

        d_inner = R_01'*p_06-P(:,2) - rot(H(:,2), theta_14)*P(:,5);
        d = norm(d_inner);
        [theta_3, theta_3_is_LS] = sp_3(-P(:,4), P(:,3), H(:,2), d);

        for i_q3 = 1:length(theta_3)
            q_3 = theta_3(i_q3);
            [q_2, q_2_is_LS] = sp_1(P(:,3) + rot(H(:,2), q_3)*P(:,4), d_inner, H(:,2));
            q_4 = wrapToPi(theta_14 - q_2 - q_3);

            q_i = [q_1; q_2; q_3; q_4; q_5; q_6];
            Q = [Q q_i];
            is_LS_vec = [is_LS_vec theta1_is_ls||theta5_is_ls||theta_14_is_LS||theta_3_is_LS||q_2_is_LS||q_6_is_LS];
        end
    end
end
end

%% ====== Subproblems ======
function [theta, is_LS] = sp_1(p1,p2,k)
KxP = cross(k,p1); A = [KxP -cross(k,KxP)];
x = A'*p2;
theta = atan2(x(1), x(2));
is_LS = abs(norm(p1)-norm(p2))>1e-8 || abs(dot(k,p1)-dot(k,p2))>1e-8;
end

function [theta,is_LS] = sp_3(p1,p2,k,d)
[theta,is_LS] = sp_4(p2,p1,k,0.5*(dot(p1,p1)+dot(p2,p2)-d^2));
end

function [theta,is_LS] = sp_4(h,p,k,d)
A_11 = cross(k,p); A_1 = [A_11 -cross(k,A_11)];
A = h'*A_1; b = d - h'*k*(k'*p); norm_A_2 = dot(A,A);
x_ls_tilde = A_1'*(h*b);

if norm_A_2 > b^2
    xi = sqrt(norm_A_2-b^2);
    x_N_prime_tilde = [A(2); -A(1)];
    sc_1 = x_ls_tilde + xi*x_N_prime_tilde;
    sc_2 = x_ls_tilde - xi*x_N_prime_tilde;
    theta = [atan2(sc_1(1), sc_1(2)) atan2(sc_2(1), sc_2(2))];
    is_LS = false;
else
    theta = atan2(x_ls_tilde(1), x_ls_tilde(2));
    is_LS = true;
end
end

%% ====== FK Function ======
function [R,p] = fwdkin(kin,theta)
p = kin.P(:,1); R = eye(3);
for i = 1:numel(kin.joint_type)
    if kin.joint_type(i)==0 || kin.joint_type(i)==2
        R = R*rot(kin.H(:,i),theta(i));
    elseif kin.joint_type(i)==1 || kin.joint_type(i)==3
        p = p + R*kin.H(:,i)*theta(i);
    end
    p = p + R*kin.P(:,i+1);
end
end

%% ====== Rotation Helper ======
function R = rot(k,theta)
k = k/norm(k); R = eye(3)+sin(theta)*hat(k)+(1-cos(theta))*hat(k)*hat(k);
end

function khat = hat(k)
khat=[0 -k(3) k(2); k(3) 0 -k(1); -k(2) k(1) 0];
end

function a = wrapToPi(a)
a = mod(a+pi,2*pi)-pi;
end
