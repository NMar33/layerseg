from io import BytesIO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ADDITONAL_SPACE = 0.1

def plot_imgs_with_mask(img, masks, labels, dpi, fig_sz, suptitle=None, show_img=True, img_cmap="rgb", masks_cmap=["spring"], interpolation="none"):
  img_h, img_w = img.shape[:2]
  ratio = img_h / img_w
  show_img = 1 if show_img else 0
  # extr_space = 0.1 if suptitle is not None else 0
  fig, ax = plt.subplots(1, len(masks) + show_img, figsize=((len(masks) + show_img) * fig_sz, fig_sz * ratio * (1 + ADDITONAL_SPACE * (1 / ratio))), dpi=dpi)
  # if suptitle is not None:
  #   fig.suptitle(suptitle, fontsize=16, y=1.05)

  # Convert ax to a numpy array if it's not an array
  if not isinstance(ax, np.ndarray):
      ax = np.array([ax])
 
  if show_img:
    if img_cmap == "rgb" and len(img.shape) == 3:
      ax[0].imshow(img, interpolation=interpolation)
    else:
      img_cmap = "gray" if img_cmap == "rgb" else img_cmap
      ax[0].imshow(img, cmap=img_cmap, interpolation=interpolation)
    ax[0].set_title(labels[0])

  for i in range(len(masks)):
    # Manual alpha compositing: blend original image with colormap overlay
    # This avoids matplotlib's two-layer imshow compositing issues
    img_float = img.astype(np.float64) / 255.0 if img.dtype == np.uint8 else img.astype(np.float64)
    if img_float.ndim == 2:
      img_float = np.stack([img_float] * 3, axis=-1)

    mask_cmap_name = masks_cmap[i] if len(masks_cmap) > i else masks_cmap[-1]
    cmap = plt.get_cmap(mask_cmap_name)
    overlay_rgb = cmap(np.zeros_like(masks[i]))[:, :, :3]
    alpha_mask = (masks[i] * 0.5)[:, :, np.newaxis]
    blended = img_float * (1 - alpha_mask) + overlay_rgb * alpha_mask
    blended = np.clip(blended, 0, 1)

    ax[i + show_img].imshow(blended, interpolation=interpolation)
    ax[i + show_img].set_title(labels[i + 1])

  st = fig.suptitle(suptitle, fontsize=16)

  buf = BytesIO()
  if suptitle is not None:
    st = fig.suptitle(suptitle, fontsize=16)
    plt.savefig(buf, bbox_extra_artists=[st], bbox_inches='tight', format='png')
    # plt.savefig(buf, bbox_extra_artists=[st], format='png')
  else:
    plt.savefig(buf, format='png')
  buf.seek(0)
  result_image = Image.open(buf).convert("RGB")
  result_image = np.array(result_image)
  plt.close()
  buf.close()
  return result_image

def plot_imgs(imgs, labels, dpi, fig_sz, suptitle=None, img_cmaps=["gray"], interpolation="none"):
  img_h, img_w = imgs[0].shape[:2]
  ratio = img_h / img_w
  fig, ax = plt.subplots(1, len(imgs), figsize=((len(imgs)) * fig_sz, fig_sz * ratio * (1 + ADDITONAL_SPACE * (1 / ratio))), dpi=dpi)
  # if suptitle is not None:
  #   fig.suptitle(suptitle, fontsize=16, y=1.05)
    
  for i, img in enumerate(imgs):
    img_cmap = img_cmaps[i] if len(img_cmaps) > i else img_cmaps[-1]
    if img_cmap == "rgb" and len(img.shape) == 3:
      ax[i].imshow(img, interpolation=interpolation)
    else:
      img_cmap = "gray" if img_cmap == "rgb" else img_cmap
      ax[i].imshow(img, cmap=img_cmap, interpolation=interpolation)
    ax[i].set_title(labels[i])

  

  buf = BytesIO()
  if suptitle is not None:
    st = fig.suptitle(suptitle, fontsize=16)
    plt.savefig(buf, bbox_extra_artists=[st], bbox_inches='tight', format='png')
    # plt.savefig(buf, bbox_extra_artists=[st], format='png')
  else:
    plt.savefig(buf, format='png')
  buf.seek(0)
  result_image = Image.open(buf).convert("RGB")
  result_image = np.array(result_image)
  plt.close()
  buf.close()
  return result_image