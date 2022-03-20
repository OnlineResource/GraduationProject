function spots_t = spots_move(spots_t,spots_ini)

spots_t.h_position = spots_t.h_position + spots_t.velocity .* sin(spots_t.direction);
spots_t.w_position = spots_t.w_position + spots_t.velocity .* cos(spots_t.direction);

% velocity_hold_frame ++，记录速度超过高阈值后的帧数：
spots_t.velocity_hold_frame = spots_t.velocity_hold_frame ...
    + (spots_t.velocity >= spots_ini.velocity_h_threshold);

% velocity_hold_frame 标记为 -1 ，意味着超过了高阈值若干帧，此时必定减速
spots_t.velocity_hold_frame(spots_t.velocity_hold_frame >= spots_ini.velocity_hold) = -1;

% velocity_hold_frame 标记为 -2 ，意味着低于低阈值，此时可随机加减速(默认状态)
spots_t.velocity_hold_frame(spots_t.velocity < spots_ini.velocity_l_threshold) = -2;

% velocity_hold_frame 标记为 0 ，意味着从低速越过低阈值，此时必定加速
spots_t.velocity_hold_frame((spots_t.velocity > spots_ini.velocity_l_threshold &...
    spots_t.velocity_hold_frame == -2)) = 0;

end
