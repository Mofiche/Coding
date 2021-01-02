public class NotValidExeption extends Exception {

    public NotValidExeption(){
        super("Noten liegen nicht zwischen 0 und 15");
    }

    public NotValidExeption(String fehler){
        super(fehler);
    }

}
