function spot_one = spots_move_in(size_frame,spots_ini,gen_edge)
if nargin == 2
    gen_edge = 5;
end
[H,W,~] = deal(size_frame(1),size_frame(2),size_frame(3));
% 个数
% 随机生成位置(边缘) /pixel
side_rand = randi([0 3]);
switch side_rand
    case 0
        spot_one.h_position = gen_edge*rand();
        spot_one.w_position = W*rand();
    case 1
        spot_one.h_position = H - gen_edge*rand();
        spot_one.w_position = W*rand();
    case 2
        spot_one.h_position = H*rand();
        spot_one.w_position = gen_edge*rand();
    case 3
        spot_one.h_position = H*rand();
        spot_one.w_position = W - gen_edge*rand();
end

% 随机生成大小(直径) /pixel
spot_one.size = spots_ini.size_range(1) ...
    + (spots_ini.size_range(2) - spots_ini.size_range(1))*rand();
% 随机生成中心亮度 /16bits
spot_one.intensity = spots_ini.intensity_range(1) ...
    + (spots_ini.intensity_range(2) - spots_ini.intensity_range(1))*rand();
% 随机生成当前速度 /pixel
spot_one.velocity = spots_ini.velocity_range(1) ...
    + (spots_ini.velocity_range(2) - spots_ini.velocity_range(1))*rand();
% 随机生成当前运动方向θ∈[0,2*pi],以右侧为x轴，顺时针
spot_one.direction = 2*pi*rand();
% 速度超过 velocity_h_threshold 的帧数计数器
spot_one.velocity_hold_frame = 0;
end