import tempfile
from datetime import datetime, timedelta
from glob import glob
from os.path import exists
from time import sleep

from numpy import isnan, array, arange, meshgrid, ma, zeros
from dask.distributed import as_completed
from distributed import Client, LocalCluster
from py_eddy_tracker.dataset.grid import RegularGridDataset, GridCollection


class LAVDGrid(RegularGridDataset):
    def init_speed_coef(self, uname="u", vname="v"):
        """Hack to be able to identify eddy with LAVD field"""
        self._speed_ev = self.grid("lavd")

    @classmethod
    def from_(cls, x, y, z):
        z.mask += isnan(z.data)
        datas = dict(lavd=z, lon=x, lat=y)
        return cls.with_array(coordinates=("lon", "lat"), datas=datas, centered=True)


def identification_(filenames, err=70, forward=True):
    date = filenames[0].split('_')[-2]
    year = int(date[:4])
    f_b = 'forward' if forward else 'backward'
    path_out = f"data/eddies_med_lavd_allsat_dt2018_err{err:03}_{f_b}/{year}"
    forward_out = f"{path_out}/A_C_lavd_err{err:03}_{date}.nc"
    if not exists(forward_out):
        ref = datetime(1950, 1, 1)
        dates = array([(datetime.strptime(filename.split('_')[-2], '%Y%m%d') - ref).days for filename in filenames])
        c = GridCollection.from_netcdf_list(
            filenames,
            dates,
            "longitude",
            "latitude",
            heigth="adt",
        )

        # Add vorticity at each time step
        for g in c:
            u_y = g.compute_stencil(g.grid("u"), vertical=True)
            v_x = g.compute_stencil(g.grid("v"))
            g.vars["vort"] = v_x - u_y

        # Time properties, for example with advection only 25 days
        nb_days, step_by_day = 25, 6
        nb_time = step_by_day * nb_days
        kw_p = dict(nb_step=1, time_step=86400 / step_by_day)
        t0 = dates[0]
        t0_grid = c[t0]
        # Geographic properties, we use a coarser resolution for time consuming reasons
        step = 1 / 32.0
        x_g, y_g = arange(-6, 36, step), arange(30, 46, step)
        x0, y0 = meshgrid(x_g, y_g)
        original_shape = x0.shape
        x0, y0 = x0.reshape(-1), y0.reshape(-1)
        # Get all particles in defined area
        m = ~isnan(t0_grid.interp("vort", x0, y0))
        x0, y0 = x0[m], y0[m]
        print(f"{x0.size} particles advected")
        # Gridded mask
        m = m.reshape(original_shape)
        
        lavd = zeros(original_shape)
        lavd_ = lavd[m]
        p = c.advect(x0.copy(), y0.copy(), "u", "v", t_init=t0, **kw_p)
        for _ in range(nb_time):
            t, x, y = p.__next__()
            lavd_ += abs(c.interp("vort", t / 86400.0, x, y))
        lavd[m] = lavd_ / nb_time
        # Put LAVD result in a standard py eddy tracker grid
        lavd_forward = LAVDGrid.from_(x_g, y_g, ma.array(lavd, mask=~m).T)

        kw_ident = dict(
            force_speed_unit="m/s",
            force_height_unit="m",
            pixel_limit=(40, 200000),
            date=ref + timedelta(int(t0)),
            uname=None,
            vname=None,
            grid_height="lavd",
            shape_error=70,
            step=1e-6,
            nb_step_to_be_mle=0,
            sampling=20,
            sampling_method="visvalingam",
        )
        forward, trash = lavd_forward.eddy_identification(**kw_ident)
        out_name = f"%(path)s/A_C_lavd_err{err:03}_{date}.nc"
        forward.write_file(path=path_out, filename=out_name, format="NETCDF3_64BIT_DATA")

    sleep(0.01)
    return filename


if __name__ == "__main__":
    log_dask = tempfile.tempdir
    kwargs = dict(
        local_directory=log_dask,
        log_directory=log_dask,
    )
    if True:
        import multiprocessing

        cluster = LocalCluster(
            int(multiprocessing.cpu_count()) - 2,
            threads_per_worker=1,
            memory_limit="3GB",
        )
        client = Client(cluster)
    else:
        import dask_jobqueue

        cluster = dask_jobqueue.SGECluster(
            cores=1,
            memory="3GB",
            walltime=10000,
            resource_spec="mem_total=3G",
            **kwargs,
        )
        cluster.adapt(minimum_jobs=2, maximum_jobs=400)
        client = Client(cluster)

    print("dashboard : ", client.scheduler_info()["services"]["dashboard"])
    regexps = [
        "../work/data/cmems_dt2018_med_allsat/*/*/dt_med_allsat_phy_l4_*.nc"
    ]
    filenames = glob(regexps[0])
    filenames.sort()
    futures = list()
    for i, filename in enumerate(filenames):
        if i > 8 * 366:
            break
        r = client.submit(identification_, filenames=filenames[i:i+30])
        futures.append(r)
    for future in as_completed(futures):
        if future.status == "error":
            print("---------------")
            print(future.key)
            print(future.exception())
            print("+++++++++++++++")
        else:
            # print(future.result())
            pass
    sleep(5)
    cluster.close()
    client.close()
