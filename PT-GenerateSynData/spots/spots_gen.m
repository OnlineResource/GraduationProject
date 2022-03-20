function spots = spots_gen(size_frame,spots_ini)
[H,W,T] = deal(size_frame(1),size_frame(2),size_frame(3));
% 初始时刻的个数
N = randi(spots_ini.quantity_range);
spots_ini.ID_total = spots_ini.ID_total + N;
% 帧序号
spots.spots_num = N;
% ID
spots.ID = (1:N).';
% 随机生成位置 /pixel
spots.h_position = H*rand(N,1);
spots.w_position = W*rand(N,1);
% 随机生成大小(直径) /pixel
spots.size = spots_ini.size_range(1) ...
    + (spots_ini.size_range(2) - spots_ini.size_range(1))*rand(N,1);
% 随机生成中心亮度 /16bits
spots.intensity = spots_ini.intensity_range(1) ...
    + (spots_ini.intensity_range(2) - spots_ini.intensity_range(1))*rand(N,1);
% 随机生成当前速度 /pixel
spots.velocity = spots_ini.velocity_range(1) ...
    + (spots_ini.velocity_range(2) - spots_ini.velocity_range(1))*rand(N,1);
% 随机生成当前运动方向θ∈[0,2*pi],以右侧为x轴，顺时针
spots.direction = 2*pi*rand(N,1);
% 速度超过 velocity_h_threshold 的帧数计数器
spots.velocity_hold_frame = zeros(N,1);

% 完成 T 帧迭代
for t = 2:T
    [spots,spots_ini] = spots_evo(spots,size_frame,spots_ini,t);
end

end