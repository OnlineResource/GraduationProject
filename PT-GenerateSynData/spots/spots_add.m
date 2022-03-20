function spots_t = spots_add(spots_t,spot_one,new_ID)
spots_t.ID = [spots_t.ID;new_ID + 1];
spots_t.h_position = [spots_t.h_position;spot_one.h_position];
spots_t.w_position = [spots_t.w_position;spot_one.w_position];
spots_t.size = [spots_t.size;spot_one.size];
spots_t.intensity = [spots_t.intensity;spot_one.intensity];
spots_t.velocity = [spots_t.velocity;spot_one.velocity];
spots_t.direction = [spots_t.direction;spot_one.direction];
spots_t.velocity_hold_frame = [spots_t.velocity_hold_frame;spot_one.velocity_hold_frame];
end