def notify(sprites, event, info):
    for key, sprite in sprites.items():
        if hasattr(sprite, "on_" + event):
            sprites[key] = getattr(sprite, "on_" + event)(sprites, info)
    return sprites
