/// SAH-binned BVH builder (port of scene-loader.js buildBVH)
use glam::Vec3;

const MAX_LEAF_SIZE: usize = 4;
const SAH_BINS: usize = 12;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BVHNode {
    pub aabb_min: [f32; 3],
    pub left_first: u32,
    pub aabb_max: [f32; 3],
    pub tri_count: u32,
}

pub struct BVH {
    pub nodes: Vec<BVHNode>,
    pub sorted_tri_data: Vec<[u32; 4]>, // [v0, v1, v2, mat_id] per tri, reordered
}

pub fn build_bvh(
    positions: &[f32],      // flat xyz
    indices: &[u32],        // flat i0,i1,i2 per tri
    tri_mat_ids: &[u32],    // mat_id per tri
    on_progress: impl Fn(&str),
) -> BVH {
    let tri_count = indices.len() / 3;
    on_progress(&format!("Building BVH for {} triangles...", tri_count));

    // Compute per-triangle AABB and centroid
    let mut centroids = vec![Vec3::ZERO; tri_count];
    let mut t_min = vec![Vec3::splat(1e30); tri_count];
    let mut t_max = vec![Vec3::splat(-1e30); tri_count];

    for i in 0..tri_count {
        let i0 = indices[i * 3] as usize;
        let i1 = indices[i * 3 + 1] as usize;
        let i2 = indices[i * 3 + 2] as usize;
        let a = Vec3::new(positions[i0 * 3], positions[i0 * 3 + 1], positions[i0 * 3 + 2]);
        let b = Vec3::new(positions[i1 * 3], positions[i1 * 3 + 1], positions[i1 * 3 + 2]);
        let c = Vec3::new(positions[i2 * 3], positions[i2 * 3 + 1], positions[i2 * 3 + 2]);
        centroids[i] = (a + b + c) / 3.0;
        t_min[i] = a.min(b).min(c);
        t_max[i] = a.max(b).max(c);
    }

    // Triangle order (will be reordered by BVH build)
    let mut tri_order: Vec<u32> = (0..tri_count as u32).collect();

    // Pre-allocate nodes
    let mut nodes: Vec<BVHNode> = Vec::with_capacity(tri_count * 2);
    nodes.push(BVHNode {
        aabb_min: [0.0; 3],
        left_first: 0,
        aabb_max: [0.0; 3],
        tri_count: 0,
    });

    fn surface_area(mn: Vec3, mx: Vec3) -> f32 {
        let d = mx - mn;
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    fn compute_aabb(
        tri_order: &[u32],
        start: usize,
        end: usize,
        t_min: &[Vec3],
        t_max: &[Vec3],
    ) -> (Vec3, Vec3) {
        let mut mn = Vec3::splat(1e30);
        let mut mx = Vec3::splat(-1e30);
        for i in start..end {
            let t = tri_order[i] as usize;
            mn = mn.min(t_min[t]);
            mx = mx.max(t_max[t]);
        }
        (mn, mx)
    }

    // Recursive build
    fn subdivide(
        node_idx: usize,
        start: usize,
        end: usize,
        nodes: &mut Vec<BVHNode>,
        tri_order: &mut [u32],
        centroids: &[Vec3],
        t_min: &[Vec3],
        t_max: &[Vec3],
        depth: u32,
    ) {
        let count = end - start;
        let (mn, mx) = compute_aabb(tri_order, start, end, t_min, t_max);
        nodes[node_idx].aabb_min = mn.to_array();
        nodes[node_idx].aabb_max = mx.to_array();

        if count <= MAX_LEAF_SIZE || depth > 32 {
            nodes[node_idx].left_first = start as u32;
            nodes[node_idx].tri_count = count as u32;
            return;
        }

        // Find best SAH split
        let parent_sa = surface_area(mn, mx);
        let leaf_cost = count as f32;
        let mut best_cost = leaf_cost;
        let mut best_axis = usize::MAX;
        let mut best_split = 0usize;

        // Centroid bounds
        let mut cmin = Vec3::splat(1e30);
        let mut cmax = Vec3::splat(-1e30);
        for i in start..end {
            let c = centroids[tri_order[i] as usize];
            cmin = cmin.min(c);
            cmax = cmax.max(c);
        }

        let mut bin_count = [0i32; SAH_BINS];
        let mut bin_min = [Vec3::splat(1e30); SAH_BINS];
        let mut bin_max = [Vec3::splat(-1e30); SAH_BINS];

        for axis in 0..3 {
            let ext = cmax[axis] - cmin[axis];
            if ext < 1e-8 {
                continue;
            }
            let scale = SAH_BINS as f32 / ext;

            // Reset bins
            bin_count.fill(0);
            bin_min.fill(Vec3::splat(1e30));
            bin_max.fill(Vec3::splat(-1e30));

            for i in start..end {
                let t = tri_order[i] as usize;
                let b = ((centroids[t][axis] - cmin[axis]) * scale) as usize;
                let b = b.min(SAH_BINS - 1);
                bin_count[b] += 1;
                bin_min[b] = bin_min[b].min(t_min[t]);
                bin_max[b] = bin_max[b].max(t_max[t]);
            }

            // Sweep right to left
            let mut right_count = [0i32; SAH_BINS];
            let mut right_min = [Vec3::splat(1e30); SAH_BINS];
            let mut right_max = [Vec3::splat(-1e30); SAH_BINS];
            let mut rc = 0i32;
            let mut rmn = Vec3::splat(1e30);
            let mut rmx = Vec3::splat(-1e30);
            for i in (1..SAH_BINS).rev() {
                rc += bin_count[i];
                rmn = rmn.min(bin_min[i]);
                rmx = rmx.max(bin_max[i]);
                right_count[i] = rc;
                right_min[i] = rmn;
                right_max[i] = rmx;
            }

            // Sweep left to right
            let mut lc = 0i32;
            let mut lmn = Vec3::splat(1e30);
            let mut lmx = Vec3::splat(-1e30);
            for i in 0..SAH_BINS - 1 {
                lc += bin_count[i];
                lmn = lmn.min(bin_min[i]);
                lmx = lmx.max(bin_max[i]);
                let cost = lc as f32 * surface_area(lmn, lmx)
                    + right_count[i + 1] as f32 * surface_area(right_min[i + 1], right_max[i + 1]);
                let cost = cost / parent_sa;
                if cost < best_cost {
                    best_cost = cost;
                    best_axis = axis;
                    best_split = i + 1;
                }
            }
        }

        // No good split found → make leaf
        if best_axis == usize::MAX {
            nodes[node_idx].left_first = start as u32;
            nodes[node_idx].tri_count = count as u32;
            return;
        }

        // Partition
        let ext = cmax[best_axis] - cmin[best_axis];
        let scale = SAH_BINS as f32 / ext;
        let mut i = start;
        let mut j = end - 1;
        while i <= j {
            let t = tri_order[i] as usize;
            let b = ((centroids[t][best_axis] - cmin[best_axis]) * scale) as usize;
            let b = b.min(SAH_BINS - 1);
            if b < best_split {
                i += 1;
            } else {
                tri_order.swap(i, j);
                if j == 0 {
                    break;
                }
                j -= 1;
            }
        }

        let mid = i;
        if mid == start || mid == end {
            nodes[node_idx].left_first = start as u32;
            nodes[node_idx].tri_count = count as u32;
            return;
        }

        // Create children
        let left_idx = nodes.len();
        nodes.push(BVHNode {
            aabb_min: [0.0; 3],
            left_first: 0,
            aabb_max: [0.0; 3],
            tri_count: 0,
        });
        nodes.push(BVHNode {
            aabb_min: [0.0; 3],
            left_first: 0,
            aabb_max: [0.0; 3],
            tri_count: 0,
        });

        nodes[node_idx].left_first = left_idx as u32;
        nodes[node_idx].tri_count = 0;

        subdivide(
            left_idx,
            start,
            mid,
            nodes,
            tri_order,
            centroids,
            t_min,
            t_max,
            depth + 1,
        );
        subdivide(
            left_idx + 1,
            mid,
            end,
            nodes,
            tri_order,
            centroids,
            t_min,
            t_max,
            depth + 1,
        );
    }

    subdivide(
        0,
        0,
        tri_count,
        &mut nodes,
        &mut tri_order,
        &centroids,
        &t_min,
        &t_max,
        0,
    );

    on_progress(&format!(
        "BVH built: {} nodes for {} triangles",
        nodes.len(),
        tri_count
    ));

    // Reorder triangle data by BVH order
    let mut sorted_tri_data = vec![[0u32; 4]; tri_count];
    for i in 0..tri_count {
        let t = tri_order[i] as usize;
        sorted_tri_data[i] = [
            indices[t * 3],
            indices[t * 3 + 1],
            indices[t * 3 + 2],
            tri_mat_ids[t],
        ];
    }

    BVH {
        nodes,
        sorted_tri_data,
    }
}
