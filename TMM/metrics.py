import os
import torch

#==========================
# Depth Prediction Metrics
#==========================

def compute_depth_metrics(gt, pred, mask=None, median_align=False):
    """Computation of metrics between predicted and ground truth depths
    """

    if mask is None:
        mask = gt > 0
    
    gt = gt[mask]
    pred = pred[mask]

    gt[gt<0.1] = 0.1
    pred[pred<0.1] = 0.1
    gt[gt>10] = 10
    gt[pred>10] = 10

    if median_align:
        pred *= torch.median(gt) / torch.median(pred)

    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log10(gt) - torch.log10(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_ = torch.mean(torch.abs(gt - pred))

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    log10 = torch.mean(torch.abs(torch.log10(pred/gt)))

    return abs_, abs_rel, sq_rel, rmse, rmse_log, log10, a1, a2, a3

#==========================
# Normal Prediction Metrics
#==========================

def compute_normal_metrics(gt, pred):
    """
    Computation of metrics between predicted and ground truth normals
    """

    # Normalize the vectors to unit vectors because cosine similarity requires unit vectors
    gt_norm = torch.nn.functional.normalize(gt, p=2, dim=1)
    pred_norm = torch.nn.functional.normalize(pred, p=2, dim=1)

    # Compute cosine similarity between ground truth and predicted normals
    cos_sim = (gt_norm * pred_norm).sum(dim=1)  # Element-wise multiplication and sum across channel dimension
    cos_sim = torch.clamp(cos_sim, -1, 1)  # Clamp values to ensure numerical stability for acos

    # Calculate angle in degrees between predictions and true values
    angle_errors = torch.acos(cos_sim) * (180.0 / torch.pi)  # Convert from radians to degrees

    # Metrics
    mean_angle_error = angle_errors.mean()  # Mean angle error
    median_angle_error = angle_errors.median()  # Median angle error
    std_angle_error = angle_errors.std()  # Standard deviation of the angle errors

    # Percentage of angles less than 11.25 degrees, 22.5 degrees, and 30 degrees
    a1 = (angle_errors < 11.25).float().mean()
    a2 = (angle_errors < 22.5).float().mean()
    a3 = (angle_errors < 30).float().mean()

    return mean_angle_error, median_angle_error, std_angle_error, a1, a2, a3


    return results

# From https://github.com/fyu/drn
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.vals = []
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.vals.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def to_dict(self):
        return {
            'val': self.val,
            'sum': self.sum,
            'count': self.count,
            'avg': self.avg
        }

    def from_dict(self, meter_dict):
        self.val = meter_dict['val']
        self.sum = meter_dict['sum']
        self.count = meter_dict['count']
        self.avg = meter_dict['avg']


#  class Evaluator(object):

#     def __init__(self, median_align=False):

#         self.median_align = median_align
#         # Error and Accuracy metric trackers
#         self.metrics = {}
#         self.metrics["err/abs_"] = AverageMeter()
#         self.metrics["err/abs_rel"] = AverageMeter()
#         self.metrics["err/sq_rel"] = AverageMeter()
#         self.metrics["err/rms"] = AverageMeter()
#         self.metrics["err/log_rms"] = AverageMeter()
#         self.metrics["err/log10"] = AverageMeter()
#         self.metrics["acc/a1"] = AverageMeter()
#         self.metrics["acc/a2"] = AverageMeter()
#         self.metrics["acc/a3"] = AverageMeter()

#     def reset_eval_metrics(self):
#         """
#         Resets metrics used to evaluate the model
#         """
#         self.metrics["err/abs_"].reset()
#         self.metrics["err/abs_rel"].reset()
#         self.metrics["err/sq_rel"].reset()
#         self.metrics["err/rms"].reset()
#         self.metrics["err/log_rms"].reset()
#         self.metrics["err/log10"].reset()
#         self.metrics["acc/a1"].reset()
#         self.metrics["acc/a2"].reset()
#         self.metrics["acc/a3"].reset()

#     def compute_eval_metrics(self, gt_depth, pred_depth, mask):
#         """
#         Computes metrics used to evaluate the model
#         """
#         N = gt_depth.shape[0]

#         abs_, abs_rel, sq_rel, rms, rms_log, log10, a1, a2, a3 = \
#             compute_depth_metrics(gt_depth, pred_depth, mask, self.median_align)

#         self.metrics["err/abs_"].update(abs_, N)
#         self.metrics["err/abs_rel"].update(abs_rel, N)
#         self.metrics["err/sq_rel"].update(sq_rel, N)
#         self.metrics["err/rms"].update(rms, N)
#         self.metrics["err/log_rms"].update(rms_log, N)
#         self.metrics["err/log10"].update(log10, N)
#         self.metrics["acc/a1"].update(a1, N)
#         self.metrics["acc/a2"].update(a2, N)
#         self.metrics["acc/a3"].update(a3, N)

#     def print(self, dir=None):
#         avg_metrics = []
#         avg_metrics.append(self.metrics["err/abs_"].avg)
#         avg_metrics.append(self.metrics["err/abs_rel"].avg)
#         avg_metrics.append(self.metrics["err/sq_rel"].avg)
#         avg_metrics.append(self.metrics["err/rms"].avg)
#         avg_metrics.append(self.metrics["err/log_rms"].avg)
#         avg_metrics.append(self.metrics["err/log10"].avg)
#         avg_metrics.append(self.metrics["acc/a1"].avg)
#         avg_metrics.append(self.metrics["acc/a2"].avg)
#         avg_metrics.append(self.metrics["acc/a3"].avg)

#         print("\n  "+ ("{:>9} | " * 9).format("abs_", "abs_rel", "sq_rel", "rms", "rms_log", "log10", "a1", "a2", "a3"))
#         print(("&  {: 8.5f} " * 9).format(*avg_metrics))

#         if dir is not None:
#             file = os.path.join(dir, "result.txt")
#             with open(file, 'w') as f:
#                 print("\n  " + ("{:>9} | " * 9).format("abs_", "abs_rel", "sq_rel", "rms", "rms_log",
#                                                       "log10", "a1", "a2", "a3"), file=f)
#                 print(("&  {: 8.5f} " * 9).format(*avg_metrics), file=f)

class Evaluator(object):

    def __init__(self):
        # Error and Accuracy metric trackers for normals
        self.metrics = {}
        self.metrics["mean_angle_error"] = AverageMeter()
        self.metrics["median_angle_error"] = AverageMeter()
        self.metrics["std_angle_error"] = AverageMeter()
        self.metrics["acc/a1"] = AverageMeter()
        self.metrics["acc/a2"] = AverageMeter()
        self.metrics["acc/a3"] = AverageMeter()

    def reset_eval_metrics(self):
        """
        Resets metrics used to evaluate the model
        """
        for metric in self.metrics.values():
            metric.reset()

    def compute_eval_metrics(self, gt_normals, pred_normals):
        """
        Computes metrics used to evaluate the model for normal vectors
        """
        N = gt_normals.shape[0]

        results = compute_normal_metrics(gt_normals, pred_normals)

        self.metrics["mean_angle_error"].update(results[0], N)
        self.metrics["median_angle_error"].update(results[1], N)
        self.metrics["std_angle_error"].update(results[2], N)
        self.metrics["acc/a1"].update(results[3], N)
        self.metrics["acc/a2"].update(results[4], N)
        self.metrics["acc/a3"].update(results[5], N)

    def print(self, dir=None):
        """
        Print or save the evaluation results.
        """
        avg_metrics = [
            self.metrics["mean_angle_error"].avg,
            self.metrics["median_angle_error"].avg,
            self.metrics["std_angle_error"].avg,
            self.metrics["acc/a1"].avg,
            self.metrics["acc/a2"].avg,
            self.metrics["acc/a3"].avg
        ]

        headers = ["mean_err", "median_err", "std_err", "a1", "a2", "a3"]
        header_string = " | ".join("{:>12}".format(header) for header in headers)
        metrics_string = " | ".join("{: 12.5f}".format(metric) for metric in avg_metrics)

        print("\n  " + header_string)
        print(metrics_string)

        if dir is not None:
            file = os.path.join(dir, "result.txt")
            with open(file, 'w') as f:
                print("\n  " + header_string, file=f)
                print(metrics_string, file=f)
