plt.figure(figsize=(10,7))

plt.plot(his.his['loss'], label='Train_loss')
plt.plot(his.his['val_loss'], label='Validation loss')

plt.title("loss")
plt.legend()
plt.show()