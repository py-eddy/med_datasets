CDF       
      obs    >   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��n��P      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P��      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���
   max       >�      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�Q��   max       @E��\)     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ����
=p    max       @vqG�z�     	�  *D   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @O�           |  3�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��`          �  4p   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��C�   max       >M��      �  5h   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�Z�   max       B,�      �  6`   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��+   max       B,��      �  7X   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >��I   max       C�r�      �  8P   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =�E�   max       C�w]      �  9H   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  :@   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          M      �  ;8   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          '      �  <0   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P�      �  =(   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���E��   max       ?�(����      �  >    speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���
   max       >�      �  ?   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>\(��   max       @E��\)     	�  @   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ����
=p    max       @vqG�z�     	�  I�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @M�           |  Sp   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�@          �  S�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         C   max         C      �  T�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�hr� Ĝ   max       ?�&�x���     `  U�            !      F      S   	      	         -   )         (                              	             '                  n   5      Z   
                     "      4   -         �      
      d         M��N��lNv��P�N�X�PA�N�kP��N�lN~~[N۠N�ZO��?O��JO�,N�3TO'x�O�A"NbN
�P�KO{��N8RN���N���N21�N$z�Oq�-O.?�O,�O�yuO�~N�]O_$�O];Oq�N��P`�.P8f|N-WPV�NὒN�lAN@��O��O�|�N�� OUuO7@LOx8O�{�O��DOHX�N���PpQHN5N?�aN�y�O�N��gN�/�N�J\���
���
�D����`B;D��;�`B<t�<49X<T��<T��<e`B<e`B<u<u<�C�<�t�<���<��
<��
<�1<���<�/<�`B<�`B<�`B<�h<�h<�<��<��<��<��=o=o=C�=C�=\)=\)=\)=�w=,1=,1=49X=<j=L��=T��=aG�=e`B=m�h=}�=��=�+=�+=�C�=�\)=�hs=���=��=��=��`=�;d>�UY[gtttig[UUUUUUUUUU_]bhnt�������tqh____��������������������
2<]ns���n]UIAL?0
geejmuz||���zsmggggECHt�����������th[OE����������������������5BZXcyyg[N7/ ��������������������������������������������������������� ��������)5BFNIB)
'/7BOht���tf^d[XOB)'vqqnq�������������zvLIOY[htu{tqh[OLLLLLL")08BO[^hlhe[OOB6+)"����������������������������������������779<@HLPH<7777777777��������������������mjiikt�����������{tm���������������������������


������4026<HTU]YUOH<444444����	���������������������������������)79BEMMN[\[LB5~�����������������~
#/5;<?BHF=<#"
$/<HY^bc^UMH<6/#[cht��������}gbab[[iikmnsz{�������zomiit�������������xuywtt��

#+-#
�����
#$))(# 
���������������������������&)%���������4-/2/7BN��������gM<4������������������������)5=;=C_gg[/���XQT[gjt�����tge[XXXX���!��������������������������!!)67:@BINSGB<;6)!vx{���������������{v*,,/:<HLRHHG=<;/****���� 
##&#
��  	"/<?=;4/%"	[UX[accmz�������zma[�������#������}~�����������������//4<HUanurnlda[UH<3/��������������������{������������������{���������������������
 #&'%#
	 ������������������	���������������������������������������������^\]aanuz{|{zynca^^^^�����������������������������������������@�L�Y�]�e�o�q�l�e�Y�T�L�A�@�@�<�@�@�@�@�������������������|�����������������������������ɼ��������|�f�=�1�@�M�M�D�Y�r��������������������������������������������������Ⱦ;Ⱦ˾ƾ�������f�Z�I�J�O�a��A�M�R�a�f�m�s�v�s�f�Z�M�A�4�(���(�4�A����+�T�P�C���Ƴ�u�\��
���C�\�uƳ����������*�.�+�*� ����������������ìùû��üùìàÓÒÓßàìììììììŹ������������������ŹŭŭŪŭųŹŹŹŹ������������������ýùìììîöù�����ſ.�;�G�`���{�y�a�X�I�G�;�.�%�����"�.�������ʼҼټռּ���������f�\�f�m�~�����tāčĚĦııįĦĥĚčā�t�l�g�i�i�m�t�-�:�=�F�J�O�G�F�:�3�-�(�'�'�-�-�-�-�-�-àäìùÿÿúùíìêàÙÓÇ�|�|ÇÏà�(�5�A�N�\�k�s�{�s�r�e�Z�N�A�5�&���&�(E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E7ECEPEVEPEFECE7E.E.E7E7E7E7E7E7E7E7E7E7�N�g�s�����������s�g�N�I�5�(����&�1�N������(�/�3�(�!�������ݿؿۿݿ��������$�"�������������A�M�Z�f�h�f�c�f�g�f�Z�M�A�>�8�<�A�A�A�A��������������������������������������&�(���������������������������������������������������������(�5�A�I�R�N�Z�g�g�Z�N�K�D�A�$����%�(�����������������������~�v�x���������������"�(�-�1�5�A�D�A�5�(�������
������������������������s�d�Z�O�P�Z�b�s���W�a�n�p�a�Y�N�H�/�#�����#�/�<�H�R�W�������	������
��������������������ݹ��������ݹϹù��������������ùϹٹݾ��ʾ׾���	��"�)��	���׾������������m�y�|����z�y�m�`�T�I�G�;�8�;�>�G�T�`�mÓàâèìñìàßÓÒÒÓÓÓÓÓÓÓÓ�������
����������àÓ�t�Y�O�M�_Ðó������/�;�H�T�\�Z�;�1�.�"����������������������������޻������������
�#�<�K�W�d�d�U�<�����ĳĚăĿ�������
�O�[�h�s�k�h�c�]�[�O�B�>�6�@�B�L�O�O�O�O�������	�
��
�������������������������	��"�.�8�8�.�"���	���	�	�	�	�	�	�	�	�r�~�����������ɺֺ�ɺ����������~�q�j�r�������ýнٽͽĽ��������y�w�������v�~��¦²·¿¿��¿²¦¦�ʾ׾�������׾ʾ��������������������H�T�a�m�n�y�z�����z�m�a�]�T�P�H�E�@�F�HŔŭŹ������������������ŹŭŠŔŋŋŏŔ�����������������������p�m�k�k�p�z�������(�4�A�M�]�f�f�`�S�A�4�(��������(������$�)�$� ���	������������������M�Z�f�o�s�s�s�f�e�Z�M�F�A�<�A�J�M�M�M�M����ּ��!�:�G�N�J�!���ּ����m�`�d��t�y�t�q�p�t�t�t�t�t�tE�E�E�E�E�E�E�E�E�E�EzEwE�E�E�E�E�E�E�E�ǮǶǶǭǪǡǔǈǄ�ǆǈǔǛǡǮǮǮǮǮD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DtDsD{D��_�l�x�������������������x�r�l�_�X�W�_�_��������������� �����������������������˻��ûлܻܻ�ܻӻлû������������������� p N D U h : j k H W  b E C  P D  ; S ( 9 Z $ ( H d 0 L _ 2 O J . ~ 3 n L 8 q { W " i v P ? V @ # I   : 6 Z g 3 ) , I g E  >    }  �  �  �  W  	;  0  �  �  �  �  �  �  �  �  I  8  1  l  �  W  �  �  G  j  �  �  �  v  d  !  �  t  M  V  �  ^  �  m  �  �  v  �  �  �  T  |  �    v  �  �  a  �  H  �  �  �  �  ̼�C��o��o<���<o=��-<��
=��<�9X<�h<�9X=o<�`B=u=m�h<ě�=o=q��<���<�/=T��=49X<�=C�=D��=C�=\)=��=D��=@�=�%=�\)=8Q�=u=@�=aG�=��>t�=�-=@�>$�=T��=ix�=L��=�o=�t�=�t�=�\)=�j=�-=�h=�G�=ě�=��w>J��=���=� �=��#>M��=�=>��B	�BjB��B&<A�j�Bq�B!��BNBnB!ьB��BƹBc�BfuB �Bv�BV�B��B��B��B�*B
��B"A�B/\B�B/~B�!Bt�B�?B�;BېB7�A���Bj�B�ZB>B �tB6�B	smB �,B�B	wmB�aBۋB�+B*��B��B$s}A�Z�A�GB��B��B�B�By�B�B��B�B��B,�Br�B��B	;�B��B��B&=�A�B��B!�HB��BʏB">_B�WB:�B��B��B ��B��B}�BÖB�`B��BTSB
΄B!�VB�,B;�B<�B��B�MB��B��B��BW�A��lBAB6�B>�B �rB@�B	O�B ��B�eB	MCB��B�uB?�B*M�BmB$KA��+A��B�B��B>�B��B��B�>B��B?�B��B,��BC�B�sA��`?ԺiA���@缮A�k�AFB�A;OBB�iA��tA�a]A�sA�r�Ac�@�Aލ@ziaA�v\A�ԧC�r�C��DA�aA���?u��A=��A�_A�حA�E�A���A�b�A��A�z+A�͌B�j>��IAT1 Ah�VAˏA�9�A���@��A�zDA���A��IA^Ww@��A XA���AQʐA��A��UA��/A8�+A��QA>�@��VA��C��B�C�Ԯ@��FA���@��9A�y+?��A��k@�#�A��AF�FA;B=�A�pA�{zA�N�A��Ac�)@��Aޔ�@|A�
�A�� C�w]C���A�nA���?�gA=0�A�{�A�tAρ�A���A�'gA�	SA�oA�B�=�E�AU�XAi-vA�yA��oA���@��A鈳AـrA�;�A]0�@$?�A!$/A�|;AP��A�~}A���A�g�A8�/A���A>mA��A��C��B@6C��Y@���A�@@�4         	   !      G      T   
      	         .   *         (                              	         !   (                  n   5   	   Z                        #      4   -         �            e      	               3      -      M               !   '                     '                                 !         !         3   -      ;               %               #            7                                                                                    !                                                      '                     %               #            %                     M��ND& Nv��OwYN�X�OL��NL9O��N��TN[q�N۠NO��O�vsO8�dO|N�3TO�2OT)wNbN
�O�w�O{��N8RN���Np9oN21�N$z�Oq�-Nɤ�O,�O�=�OaN�]O_$�O?OOq�N��O�s:P�N-WO��	NὒN�lAN@��O��O�|�N�� N�W�O�Og�Oʽ�OrN�O(^EN���O�3�N5N��N�y�O'hvN��gN�/�N�J\  P  M  �  J  �  =  �  A  �  �  �  F  r  �  e  �  �  �  |    u  p  �  �  >     �    C  �  �  �  "  �  �  ;  �  
�  o     
N  �  k    Q    �  w  �  �  �  �  �  ]  �  �  �  	�  �  �  �  ����
�e`B�D��;�o;D��=#�
<T��=y�#<u<e`B<e`B<�t�<�o<�h=o<�t�<��
<�h<��
<�1<�h<�/<�`B<�`B=C�<�h<�h<�=\)<��=C�=0 �=o=o=\)=C�=\)=�9X=0 �=�w=��-=,1=49X=<j=L��=T��=aG�=m�h=�o=�%=�7L=���=�O�=�C�=���=�hs=��-=��=��m=��`=�;d>�UY[gtttig[UUUUUUUUUUijtt�����tiiiiiiiiii��������������������" #-0<Ubehf_XUI<0#"geejmuz||���zsmgggg[Y\cpt����������thd[��������������������)57AEKMNB5)�����������������������������������������������������������������  (5MHB;);<BFOW[htw��|nh[NB;������������������LIOY[htu{tqh[OLLLLLL)%)269BO\gc[ONMB6,))����������������������������������������779<@HLPH<7777777777��������������������mjiikt�����������{tm���������������������������


������;56:<HMUWURH@<;;;;;;����	���������������������������������)79BEMMN[\[LB5��������������������
#/5;<?BHF=<#"
 '/<HV\``\UJH<9/& rmtt������������ytrriikmnsz{�������zomiit�������������xuywtt���#&*,(#
�����
#$))(# 
�������������������������������
�������423:=BNg�������t[D;4�������������������������)1341/)��XQT[gjt�����tge[XXXX���!��������������������������!!)67:@BINSGB<;6)!vx{���������������{v*,,/:<HLRHHG=<;/****����
!
	 ���	"/7;<;:0/"	\YWY\admz�������zma\������"�������������������������5106<@HUaknniaaUSH<5�������������������������������������������������������������
 #&'%#
	 ���������������������������������������������������������������^\]aanuz{|{zynca^^^^�����������������������������������������L�Y�e�e�k�e�Y�L�G�G�L�L�L�L�L�L�L�L�L�L�������������������|���������������������f�r���������������������r�k�a�\�S�Y�f�����������������������������������������s���������������������s�o�f�c�_�b�f�s�4�A�M�Y�V�M�A�4�/�*�4�4�4�4�4�4�4�4�4�4ƧƳ����������������ƧƚƁ�v�zƁƎƚƞƧ����)�&�����������������ùù��üùìàÓÒÓàáìôùùùùùùŹ������������������ŹŭŭŪŭųŹŹŹŹ������������������ùòñùü�����������ſ.�;�G�T�`�r�x�`�V�;�.�'�"��	�	���"�.�����������ʼ̼м˼ʼ�������������������āčĚĦħĪħĦěĚčā�y�t�q�t�u�zāā�-�:�=�F�J�O�G�F�:�3�-�(�'�'�-�-�-�-�-�-ÓàâìóùÿýùìàÚÕÓÇ�~�}ÇÓÓ�5�A�N�Z�b�h�g�c�Z�X�N�A�5�-�(�$��#�(�5E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E7ECEPEVEPEFECE7E.E.E7E7E7E7E7E7E7E7E7E7�N�Z�g�s������������s�g�Z�A�5�(�"�*�5�N������(�/�3�(�!�������ݿؿۿݿ��������$�"�������������A�M�Z�f�h�f�c�f�g�f�Z�M�A�>�8�<�A�A�A�A���������������������������������������&�(���������������������������������������������������������(�5�A�I�R�N�Z�g�g�Z�N�K�D�A�$����%�(�����������������������������z�~�����������"�(�-�1�5�A�D�A�5�(�������
������������������������s�h�Z�W�U�Z�f�s���/�<�H�K�R�P�H�G�<�6�/�#�����#�+�/�/�������	������
��������������������ݹ��������ݹϹù��������������ùϹٹݾ��ʾϾ׾�����	� ���׾ʾ������������m�y�|����z�y�m�`�T�I�G�;�8�;�>�G�T�`�mÓàâèìñìàßÓÒÒÓÓÓÓÓÓÓÓìù��������������������øæàÖÓÏÔì�	��/�;�H�P�S�E�;�+�"��	�������������	�������������޻������������
��#�0�<�B�I�L�E�0������������������
�O�[�h�s�k�h�c�]�[�O�B�>�6�@�B�L�O�O�O�O�������	�
��
�������������������������	��"�.�8�8�.�"���	���	�	�	�	�	�	�	�	�r�~�����������ɺֺ�ɺ����������~�q�j�r�������ýнٽͽĽ��������y�w�������v�~��¦²·¿¿��¿²¦¦�ʾ׾�������׾ʾ������������žʾ��T�a�i�m�u�z�~���z�p�m�h�a�T�S�I�D�H�L�TŔŠŭŹ����������������ŹŭŠŕŌŌŐŔ�����������������������t�m�l�l�q�{�������(�4�A�M�R�\�\�U�E�A�4�(����	�
���(���������"�'�"������������������M�Z�f�o�s�s�s�f�e�Z�M�F�A�<�A�J�M�M�M�M���ּ��������ּ������������������t�y�t�q�p�t�t�t�t�t�tE�E�E�E�E�E�E�E�E�E�E{E�E�E�E�E�E�E�E�E�ǮǶǶǭǪǡǔǈǄ�ǆǈǔǛǡǮǮǮǮǮD�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DyD{D�D�D��_�l�x�������������������x�r�l�_�X�W�_�_��������������� �����������������������˻��ûлܻܻ�ܻӻлû������������������� p K D H h ( ; = ) R  h A =  P F ! ; S " 9 Z $ ( H d 0 4 _ 0 ? J . n 3 n 5 = q / W " i v P ? K =  H  6 6 Z g - )  I g E  >  n  }    �  �  T  D  �  �  �  �  w  �  $  �  G  �  8  1  �  �  W  �  �  G  j  �  �  �  >  F  !  �  �  M  V  4  �  �  p  �  �  v  �  �  �  �  A  �  �  �  j  �  �  �  6  �  a  �  �  �  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  P  G  =  4  +  "       /  ?  O  ^  n  v  p  i  c  \  V  O  V  �  �  �  �    A  _  v  �  �  �  y  Z  1    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  `  B  e  �  �  �  *  @  H  J  A  2      �  �  �  �  �  �    a  �  �  �  �  �  �  �  �  �  t  f  Y  B  %    �  �  �  �  v  x  �  ?  u  �  �  �    3  =  ;  .  
  �  �    `  �  �  +  �  �  �  �  �  �  �  �  �  �  �  �  {  X  4    �  �  �  d  r  {  f  C    7  f  �  �    9  <  8    �  e  �  @  w  '  �  �  �  �  �  �  �  �  �  �  �  �  �  s  \  ?    �  w    �  �  �  �  ^  :    �  �  �  p  G  %    �  �  �  f  ;    �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  Y  7    �  ]  D  D  E  D  F  E  A  9  *    �  �  �    N  
  �    7  W  p  r  l  `  O  =  (    �  �  �  �  _  ;    �  �  �  ]  /  �    <  ^  u  �  �  �  n  A      �  �  �  E  �  T  C  $  �  �    :  P  Z  c  e  _  M  +  �  �  (  �  *  �  �  Z  �  �  �  �  �  �  �  �  �  �  �  ~  o  _  O  =  ,      �  �  �  �  �  �  �  �  �  x  i  Y  F  .    �  �  �  Q    �  �    N  x  �  �  �  �  l  H    �  �  x  )  �  b  �  @  u  z  |  c  J  1      �  �  �  �  �  `  4    �  �  �  �  t  T    �  �  �  �  �  �  �  �  w  c  P  =  )    �  �  �      8  q  t  q  l  g  c  \  W  M  =  &    �  �  �  t  I  �  n  p  n  i  _  O  8    �  �  �  �  �  ]  ,  �  �  �  C  �  &  �  �  �  �  �  �  �  �  �  �  }  t  l  c  [  R  J  A  8  0  �  �  �  �  �  �  �  �  �  �  �  �  z  ~  �  �  �  �  �  �  �  �      2  <  <  3  '    �  �  �  �  y  G  �  M  �       �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  �  �  �  �  �  �  �  �  o  Y  @  #    �  �  �  �  x  d  O      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  �  �    2  A  B  >  5  %    �  �  �  �  j  @    �  �  �  �  �  �  u  V  5    �  �  �  k  5  �  �  �  @  �  �  �  �  x  �  �  �  m  P  +  �  �  �  y  X  .  �  �  S  �  �  	  �  �  �  �  �  �  �  �  �  o  Z  J  7    �  �  J  �  }    k  "           �  �  �  �  �  �  c  :    �  q     �  y  .  �  M  A  %  O  �  4  �  �  �  �  f  �  f  �  k  �  <  R  �  �  �  �  �  �  �  r  T  6       �  �  �  �  u  =    �  6  ;  -        �  �  �  �  �  �  \  /  �  �  �  y  @  �  A  �  �  �  �  �  �  �  �  �  �  �  �  �  w  d  Q  >  +      	#  	�  
  
f  
�  
�  
�  
�  
�  
�  
�  
�  
�  
D  	�  	>  o  Z  �  �  *  T  i  o  h  a  U  D  -  	  �  �  �  �  P  �  s  �  (  �           	        �  �  �  �  �  �  �  �    m  [  H  	D  	�  	�  	�  	_  	�  	�  
K  
M  
<  
  	�  	c  �  I  �  �  �  �  �  �  �  �  �  x  \  @  !    �  �  �  �  �  �  }  Z  5  �  �  k  ]  O  ?  ,      �  �  �  �  w  ]  J  7  $    �  �  �    �  �  �  �  �  �  �  �  �  �  �  h  @     �   �   �   �   �  Q  H  ;  *    �  �  �  �  i  9      �  �  r  8  �  �  �    y  q  d  O  9  #       �  �  �  �  p  8  �  �  �  �  �  �  �  �  e  "  �  �  U  �  �     �  `      '    �  �  �  c  O  ^  i  9    �  �  �  �  �  �  y  �  �  �  1  �  �  �  �  �  �  �  �  �  �  �  _  +  �  �  u  (  �  ]  �  <  }  !  �  �  �  �  �  �  �  m  O  *  �  �  �  _    �    T  �    �  �  �  �  l  ]  J  3    �  �  �  8  �  �    x  �  �  �  �  �  �  �  �  �  �  �  �  �  �  L    �  �  5  �  �  �  Z  �  �  �  �  �  u  `  A    �  �  s    �  ;  �  y  �      ]  N  >  *      �  �  �  �  �  �  �  l  U  <    �  �  �  �  �  �  �  �  �  �  �  �  �  �  ,  
�  
@  
  	�  �  �  $  ^  �  �  �  �  b  ?  (    �  �  �  �  �  x  ^  H  4      �  �  �  �  �  �  h  H  '    �  �  �  S    �  �  y  A    �  	�  	�  	u  	K  	  �  �  Z    �  m    �  [  �  �    �  �  Z  Q  v  �  �  �  �  �  �  X  
  �  D  �      �  
�  �  o  �  �  �  �  �  �  q  X  =        �  �  w  F    �  �  �  s  [  �  �  �  �  n  T  <  $    �  �  �  �  �  �  o  \  B    �  �  �  y  Y  3    �  �  e  &  �  �  O    �  c    �  �  ]