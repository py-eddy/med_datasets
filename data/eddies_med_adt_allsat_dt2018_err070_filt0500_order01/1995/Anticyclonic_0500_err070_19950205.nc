CDF       
      obs    @   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�vȴ9X        �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��n   max       P�4�        �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��w   max       =        �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?}p��
>   max       @Fz�G�     
    �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��
=p�   max       @vq\(�     
   *�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @N            �  4�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�?        max       @���            5,   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��h   max       >Z�        6,   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��\   max       B/"        7,   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�{�   max       B/V        8,   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?�)�   max       C�m�        9,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?�\|   max       C�h        :,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �        ;,   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;        <,   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          )        =,   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��n   max       O�	�        >,   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��$�/   max       ?�(����        ?,   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��w   max       =        @,   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?}p��
>   max       @Fz�G�     
   A,   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��
=p�   max       @vq\(�     
   K,   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @!         max       @N            �  U,   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�?        max       @�~             U�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @W   max         @W        V�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��u��"    max       ?�(����        W�   
      �                           .      .             
   
         J                        <   
               .   
   2         +               �                                       A      P      \   N�E|OgڻPu�eNh��NUH�N+/RO?rCO��N�KCO��N��O�N��P~�O�4O;amN���N�!\N�*�N{XzNV=P�4�N��M��nO�O1x�Nb�N���N��hP�N���O�cN49GOw�YN�i$P1OSYO��O>N��O�#zNRpNNP��O5B�O�OxO�1N�~N�X�N�:�N��&Ot�%O[�Ot'N:ϬOBeIN���N��O;�^O�"tO RO���N[hO��NH���w�C����ͼ�9X��C��49X�t��ě��D����o��o��o$�  :�o;o;ě�;ě�;ě�<o<t�<t�<t�<t�<t�<49X<�t�<�t�<���<��
<��
<�1<�j<�/<�`B<�`B<�h<�=o=+=C�=C�=C�=C�=\)=t�=�P=#�
=0 �=0 �=0 �=<j=H�9=T��=T��=Y�=aG�=�%=�o=��
=�-=���=�
==�l�=.(*,0:<>FHIG<0......��������������������)5N[������tgB5-)��������������������������������������������������������������������������������GDFMO[hrtyutrh[UOGG`[[`anwz��~zna``````���������������������������������������������������������������� 


�����������#)B[gv~~zsiNB5),)(*,-/3<?HLQRQHG</,?=>==?BEN[afhgfa[NB?��������������������������������������������������������������������������������"&))5BBB75,)""""""""/*3ENt����������g[</~��������$8O]ft~��th[O=60)������
#-,'"
���!)36<BGEB6)!!!!!!!!MTTamtxqmba^UTMMMMMMNN[gotxytng^[SONNNNN#<Qpq\\^WH<>H</#�������������������������������	����;BGN[[][ONGB;;;;;;;;���������������������
#(+(#
����	"/;BKIB</
����������������������Vahlpu����������naV���*68@@:6*������������������������������������������&).5BFDDB5+)&&&&&&&&*)./<@HB</**********���������������$)'# �����������
!!
�����)*1/*)����@>@BO[ht{thehjjh[OB@[[[hhtw�����ztph\[[[rprt��������ytrrrrrr��������������������(26BKMOKD=96) (*25:A?65) 235<BNOQNHB522222222�����&046A6+��&"!)57BCEB@95+)&&&&��{�����������������#%#
���������
#�������� �������__aamz~�������zmlca_kgbhnz����������ztqkZTV]agmonmiaZZZZZZZZz����������������zxz���������������������Ľнݽ���������ݽнĽ��ýĽĽĽĽĽ��
�����
��������������������������
��)�6�J�R�Q�K�-�)������òêîù�����Ź������������������ŹŹűŹŹŹŹŹŹŹ�{ŇŔŚŝŔŇ�{�s�w�{�{�{�{�{�{�{�{�{�{��'�3�<�7�3�'�"�������������G�`�a�g�m�j�`�^�T�G�;�1�.�)�&�-�.�;�E�G���ʼּ�����޼ּʼ�����������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��[�g�p�t �t�g�[�R�R�Z�[�[�)�6�;�A�6�*�)�������"�)�)�)�)�)�)���(�A�H�M�P�N�I�A�,�����������(�5�A�C�A�8�5�(���(�(�(�(�(�(�(�(�(�(�.�G�T�m�v���������y�`�T�G�;�&����!�.D�D�EEE*E7E@E7E.E*E#EEED�D�D�D�D�D������������
������
���������������˼����üʼּ߼ּռͼʼǼ��������������������������������������x�t�z�������������������������������������������������������	��	�	���������������������������	���!����	��������������������������5�E�����������Z�A������ۿܿ���*�6�B�9�6�*����$�*�*�*�*�*�*�*�*�*�*�A�N�Z�g�i�g�\�Z�W�N�A�>�A�A�A�A�A�A�A�A�ʾ׾���׾������s�f�[�X�]�k�������ž���������)�.�.�)�����������������ÇÓÛàáàÝÓÇÄ�~ÄÇÇÇÇÇÇÇÇ�a�m�m�w�s�m�a�T�O�H�G�H�T�X�a�a�a�a�a�a����������������������������������������������$������ùìàÔÐàîíÿ����ÇÓàìùúùõìàÓÇÄÅÇÇÇÇÇÇ�Ŀѿݿ������"�!�������ݿ¿�����ƁƇƎƖƎƎƁ�|�u�u�u�vƁƁƁƁƁƁƁƁ�f�s�������������������s�f�Z�V�Y�Z�a�f����������������������ﾌ�������׾�	��.�B�;��㾾�������������"�/�/�;�G�E�<�/�"��	���������	���"�����(�A�N�Z�n�s�k�Z�N�A�(����������T�`�c�t�y�����������y�m�`�\�R�M�L�N�P�T���)�5�6�>�B�6�)������������hāčĚĦıĵİĦĚčā�h�[�K�B�E�O�[�h�(�.�4�;�8�4�(����� �(�(�(�(�(�(�(�(�����������������������������������������a�n�zÇÍÍÇÂ�z�n�i�a�U�T�H�=�H�U�[�aƧƳ������������������ƚƎƁ�u�h�uƁƎƧDoD{D�D�D�D�D�D�D�D�D�D�D�D�D�DxDkDfDdDo����������������������������������޻ܻ��������	����������ܻڻջ׻ػܼ�'�'�3�4�>�@�D�@�4�'����� �����_�l�x�����x�l�j�_�Z�S�L�S�Y�_�_�_�_�_�_�������Ľν˽ϽĽ������������}�u�z�������r�~�����������ĺպɺ��������~�r�h�[�i�r�#�0�<�E�I�R�U�U�I�>�#�
������������#�b�n�s�{�~�{�{�n�i�b�]�\�b�b�b�b�b�b�b�b�:�F�S�]�_�a�V�I�:�!���������$�7�:�$�0�=�I�M�J�I�>�=�0�'�$�����$�$�$�$�����������������������ﾾ�������������������ʾ׾�����׾ʾ�¿�������
�����������¿²«¤£¦²¿Ź������������������ŹŭŦŠŜşŠŭŴŹ�ɺֺ����������ֺɺ���������������ÓàìùüùñìàÝÓËÓÓÓÓÓÓÓÓ�����������ļż¼����������v�Y�Y�f�m���E7ECEPEUE\EPECE:E7E,E7E7E7E7E7E7E7E7E7E7 /  $ b 5 G    T _   V ; O I T ' 1 T x J n t X A N , - A @   B / 2 ` * Y N D : , ) 2 \ # B H d L  f D B s 9 ` H E 6 ! = ] \  �  �  �  �  a  O  �  C  �  >  �  �    d  �  �  @  �    �  n  �    .  r  �  �  �  �  u  �  �  ]  �  �  G  Q  1  �  �  �  i  ]  �  x  �      �  �  �    #  ^  2  �  C  �  l  W  )  o  �  q��h��1>Z���C��#�
�D��<#�
<�o<D��;��
<T��=49X;�o=@�<�9X=�P<�o<�o<�t�<T��<49X=���<49X<49X='�=<j<�`B<ě�<�/=���<��=Y�<��=#�
=+=���=#�
=��=<j='�=���=�P=#�
=m�h=e`B>9X=aG�=]/=e`B=@�=y�#=���=�hs=m�h=��T=�7L=�+=��>t�=�x�>6E�=�>P�`>$�B& �B�Bi�B �B"�B ��BcNB^DB̫B|�B��B �PBtB�B��BT�B!;�B*-�B�]B��B�B
:�B
�lBD#B8�B�YB��A��VB	�B��B!��B�*B%�B;B$�oBj+B�!BD�B/"B�BfBL|B��B�cB��B�B|B��B,�BʮB, �B�FB��B��BQ�BXB)c.B$BdB�)A��B�WA��\B��BM}B&<�BZ)BC�B:�B>B �RBB.B?sB�nB��B�,B �{BE~B��B�Bb`B!��B*?vB��B��BA�B
@B
�)BDB>hB�\B��A��CB	�B�B"\�B��B<IBjB$�}B>CB��BG�B/VB�NB@�B?8B��BB]�BƊB?�B�fB��B�:B,AB<B=QB��B< B:iB){)B$>�BB�A��B@PA�{�B�3BA�A*�|A��Aӌ`A��A�]?�)�AeG�@�	.C�m�A���A�f�A4JUA�;1AggGC�dA��6@�bF@�(�A�|}A��qA���A�M�A���A��=AJ�CA�4A�	A�p�A�Y,A�x�A�P�A��B`�ADo,A��ATh�A�k�A�{�Aj�nA�	�A�<�A7�A�C�AǇ�B�C��TA��2@�q�@�tV@���A!�:@�^A�}DA�T�@yQhB
Q2@X�AOYWA���A��@7{A��@��;C��xA+A�/�AӂKA�1oA��?�\|Ad��A ��C�hA�� A֑�A4�A�tbAhs�C�fA���@��k@���A�p�A�v!A�q�A���A�X�A�$;AIiA��PA��IA��A�v�A�~kA�[[A��BxrAEh>A[�AT��A���A��OAkUAՐ�A�|�A6��A�s�A�e�B�MC��hA�{@�t�@�i@��)A'e@N�A���A�|>@{��B
L@\[)AO:A��A�v�@9�A˫�@�C���         �         	                  .      /                         K                        <                  /      3         +               �                                       A      P      \            1                           !      #                        ;         )               -                  /      '                     !                                                      &                                             #                        #                                          )                           !                                                      &   N�E|O;��O��oNh��NUH�N+/RO��N�:BN�KCO��N��Ox�N��O�Z+O�4O0��N���Nz��N��fN{XzNV=O�o:N��M��nO�n�Oj�N$ĸN���N��hO+��NaK�O���N49GO^%~N�i$O�	�OSYO�U�O>N��O�#zNRpNNP��N�ԀO�OxO#�9N���N�vENm�@N��&Ot�%O[�Ot'N:ϬO�N���N��O0A�O�"tO ROT��N[hO��NH�    �  �  )  �    �  B  x      <  �  w  �  [    �  �  �  �  �  "  �  ?  �  �  0  �  �  �  }  �  #  E  }    �  �  |  �  �  �  �  �  �  �  �  x  �    �  R  V  �  3  %  @  ,  �    �    ��w�o=����9X��C��49X���
�D���D����o��o<T��$�  ;�`B;o;�`B;ě�<o<t�<t�<t�=8Q�<t�<t�<���<�j<��
<���<��
=8Q�<�9X<�<�/<�h<�`B=\)<�=49X=+=C�=C�=C�=C�=#�
=t�=���=,1=<j=49X=0 �=<j=H�9=T��=T��=e`B=aG�=�%=��=��
=�-=�l�=�
==�l�=.(*,0:<>FHIG<0......��������������������21359BN[gruurg[NB<52��������������������������������������������������������������������������������KFHNO[hntutsqjh[YOKK`[[`anwz��~zna``````���������������������������������������������������������������� 


�����������!)5B[gtyxtmdNB(,)(*,-/3<?HLQRQHG</,A>?=>@BEN[`eggf`[NBA��������������������������������������������������������������������������������"&))5BBB75,)""""""""XY\et�����������tgbX~��������A978=BO[cks{|zth[ODA�����
#%##
�����%#)6BDCB6)%%%%%%%%%%MTTamtxqmba^UTMMMMMMNN[gotxytng^[SONNNNN #/<EHPUVUQHH</#�������������������������������������;BGN[[][ONGB;;;;;;;;���������������������
#(+(#
��#-/<?IHFEB</����������������������vtsux������������zv���*68@@:6*������������������������������������������&).5BFDDB5+)&&&&&&&&*)./<@HB</**********������	������������$)'# �����������

�����)0.)A?BBOOP[`bc[OBAAAAAA^hjty�����wtrh^^^^^^rprt��������ytrrrrrr��������������������(26BKMOKD=96) (*25:A?65) 235<BNOQNHB522222222����%)-0)	��&"!)57BCEB@95+)&&&&��{�����������������������
#%#"
���������� �������__aamz~�������zmlca_mnz������������zyuqmZTV]agmonmiaZZZZZZZZz����������������zxz���������������������Ľнݽ���������ݽнĽ��ýĽĽĽĽĽ��������
����
�
��������������������������$�)�.�.�)� �������������������Ź������������������ŹŹűŹŹŹŹŹŹŹ�{ŇŔŚŝŔŇ�{�s�w�{�{�{�{�{�{�{�{�{�{��'�3�<�7�3�'�"�������������;�G�T�_�`�g�d�`�W�T�G�>�;�.�-�*�.�4�;�;���ʼּ�����׼ּӼʼ���������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��[�g�p�t �t�g�[�R�R�Z�[�[�)�6�;�A�6�*�)�������"�)�)�)�)�)�)����(�4�A�B�D�=�4�(��������������(�5�A�C�A�8�5�(���(�(�(�(�(�(�(�(�(�(�G�T�`�p�z���������y�`�T�G�;�*�#�"�.�;�GD�D�EEE*E7E@E7E.E*E#EEED�D�D�D�D�D������������
������
���������������˼����üʼּ߼ּռͼʼǼ���������������������������������������x�|�������������������������������������������������������	��	�	���������������������������	���!����	�����������������������(�5�N�i�x��}�s�j�Z�N�(���� � ���*�6�B�9�6�*����$�*�*�*�*�*�*�*�*�*�*�A�N�Z�g�i�g�\�Z�W�N�A�>�A�A�A�A�A�A�A�A�������ʾ׾ܾ޾־ʾ�������s�j�g�k�y����������%�(� ��������������� ��ÇÓØÞØÓÇÆÀÅÇÇÇÇÇÇÇÇÇÇ�a�m�m�w�s�m�a�T�O�H�G�H�T�X�a�a�a�a�a�a������������������������������������������������������������������������ÇÓàëìñùùùôìàÓÇÅÆÇÇÇÇ���������������ݿпɿſǿѿݿ�ƁƇƎƖƎƎƁ�|�u�u�u�vƁƁƁƁƁƁƁƁ�s���������������������s�f�Z�X�Z�\�d�s����������������������ﾘ�����׿	��'�.�1�!���׾ʾ������������"�/�/�;�G�E�<�/�"��	���������	���"���(�5�A�N�Z�c�g�`�Z�N�A�5�(������T�`�c�t�y�����������y�m�`�\�R�M�L�N�P�T���)�5�6�>�B�6�)������������hāčĚĦıĵİĦĚčā�h�[�K�B�E�O�[�h�(�.�4�;�8�4�(����� �(�(�(�(�(�(�(�(�����������������������������������������a�n�zÁÇÊÊÇ�z�n�a�X�U�P�U�U�a�a�a�aƧƳ������������������ƚƎƁ�u�h�uƁƎƧD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D}D{DwD{D�������� ���������������������������޻ܻ�������������ܻػڻۻܻܻܻܻܻܼ'�1�4�=�@�C�@�4�'�!����!�'�'�'�'�'�'�_�l�x�����x�l�j�_�Z�S�L�S�Y�_�_�_�_�_�_�������Ľν˽ϽĽ������������}�u�z�������r�~�����������ĺպɺ��������~�r�h�[�i�r�#�0�<�E�I�R�U�U�I�>�#�
������������#�b�n�s�{�~�{�{�n�i�b�]�\�b�b�b�b�b�b�b�b�:�F�S�W�_�T�G�F�:�-�!������!�'�-�:�$�0�=�I�M�J�I�>�=�0�'�$�����$�$�$�$�����������������������ﾥ���ʾ׾�����׾ʾ�����������������¿�������
�����������¿²«¤£¦²¿Ź������������������ŹŭŦŠŜşŠŭŴŹ�ֺ���������ֺɺ��������������ɺ�ÓàìùüùñìàÝÓËÓÓÓÓÓÓÓÓ�����������ļż¼����������v�Y�Y�f�m���E7ECEPEUE\EPECE:E7E,E7E7E7E7E7E7E7E7E7E7 / ! ! b 5 G    T _  V : O F T  4 T x I n t Z 5 I , -  H  B - 2 f * A N D : , ) " \  3 ( ] L  f D B b 9 ` D E 6  = ] \  �  �    �  a  O      �  >  �  �    �  �  �  @  {  �  �  n      .  �  )  X  �  �  o  �  �  ]  �  �  �  Q    �  �  �  i  ]    x  V  �  �  �  �  �    #  ^  v  �  C  �  l  W  �  o  �  q  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W  @W         �  �  �  �  �  �  �  �  �  �  �  ~  c  I  1    �  �  �  �  �  �  �  �  �  �  �  u  [  A  &    �  �  �  �  �  �  �  �  ,  u  u    �  �  �  �  �  �    X  \  x  �  
�  �  )        �            �  �  �  �  �  �  �  �  �  �  �  �  �  w  m  c  Y  N  A  4  $       �  �  �  g  5  �  �  �                    �  �  �  �  �  �  �  c  0  �  �  �  �  �  �  �  �  �  �  �  �  x  ^  ?    �  �  �  5  �  p    1  >  @  7  +    	  �  �  �  �  h  =  �  �  k  )    �  x  q  i  `  V  J  ;  (    �  �  �  }  9  �  �  0  �  m      �  �  �  �  �  �  �  |  k  Z  J  9  *        �  �  �    A  k  �  �    F  D  9  &    �  �  �  �  b  #  �  �  U  �  �       .  8  <  9  1  #    �  �  �  i  +  �  S  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  d  R  =  (     �   �  J  d  t  w  l  V  <  !  
  �  �  �  �  h    �  ,  �    �  �  �  �  �  ~  n  U  3  	  �  �  }  D  �  �  4  �  ;  �  A  K  Z  R  D  /    �  �  �  Q    �  �  V  )  �  �  �  /  7            �  �  �  �  �  �  �  �  �  �  �  i  8  5  8  �  �  �  �  �  �  �  �  �  y  i  U  @  )    �  �  D     �  �  �  �  �  �  �  �  �  �  �  s  H    �  �  �  Y  &   �   �  �  �  �  �  �  �  �  �  �  �  �  �  z  h  U  A  ,       �  �  �  �  �  �  y  p  g  _  V  M  C  9  0  &      	   �   �  �  �  �    6  M  d  }  �  �  �  i  (  �  ~    V    �   �  "                �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  n  Y  E  1       �   �  �  �    "  7  ?  >  8  +    �  �  |  )  �  �  |  7  �  q  4  g  �  �  �  �  q  \  F  *    �  �  x  '  �  G  �    4  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  d  0  /  .  ,  )           �  �  �  �  �  �  y  b  M  7  !  �  �  �  �  �  �  �  �  �  �  �  �  �  {  _  B  %         �  �  ?  �  
  o  �  �  �  �  �  |  2  �  �  "  �    �  �  �  �  �  �  �  �  �  �  y  `  D  $    �  �  �  o  J  �  �  X  l  t  y  {  z  t  l  a  R  7    �  �  �  t  @    �  �  �  �  �  �  �  �  p  _  M  <  +      �  �  �  �  �  �  �    !  "  "        �  �  �  �  �  �  r  G    �  �  l  S  E  ?  8  2  +  $          �  �  �  �  �  �  �  �  �  �  ?  w  }  u  P  $  �  �  �  X     �  t    �  .  �  �  �  .        �  �  �  �  �  �  �  |  `  K  :  #  	  �  �  �  `    t  �  �  �  �  �  �  �  �  p  9  �  �  (  �    ]  �  3  �  �  �  �  �  �  �  �  �  n  T  5    �  �  �  l  -   �   �  |  p  d  X  L  ?  1  "      �  �  �  �  �    h  9    �  �  �  �  �  j  ;    �  �  �  U    �  |  #  �  d    l  F  �  �  �  �  �  �  �  �  �  �  �  z  p  i  d  `  \  X  S  O  �  �  �    }  {  y  w  u  s  q  p  n  i  _  T  I  =  0  $  �  �  �  �  �  �  w  Q  &  �  �  �  B  �  �  _    �  C  �  �  �  p  g  b  T  M  ]  n  g  I     �  �  �  H    �  �  J  �  2  �    O  �  �  �  �  �  �  }    �  �  �  r  �  
  �  z  �  �  �  �  �  �  �  t  g  X  F  3  !           >  Z  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  i  c  e  j  o  u  q  d  R  @  /    &  4  4  $      �  �  �  �  �  �  �  �  �  �  �  |  m  _  Q  B  6  +       	  �  �  �  �  �      �  �  �  �  �  �  �  �  t  K    �  �  �  u  X  X  �  �  �  �  �  s  V  :        �  �  2  �      <  �  �  F  R  B  .      �  �  �  �  f  =    �  �  �  Q  %  �  �  �  V  H  :  ,        �  �  �  �  �  w  T  +    �  �  �  ]  W  Q  �  �  i  K  3  �  �  �  B  �  �  ^  �  �  �  �  &  �  3      �  �  �  �  Z  1    �  �  r  <  	  �  �  �  y  _  %    �  �  �  �  �  �  |  e  M  5      �  �  �  r  O  +  *  ?  <  4    �  �  �  [  %  �  �  \    �  �  A  �  �  c  ,            �  �  �  g     
�  
g  	�  	c  �  �  '  �    �  �  f  @  $    �  �  �  `  #  �  �  e  !  �  �  Y  
  q  �  �  
        �  �  �  n  %  �  e  
�  
N  	a  M  "  �  �  �  �  �  a  ?    �  �  �  �  W  -    �  �  k  /  �  �  b    �  �  �  �  �  �  L  	  �  n    ~  
�  	�  �  �  �  o  �  �  �  �  C  �  �  w  A    �  �  \  D  �  �  &  �  o    �