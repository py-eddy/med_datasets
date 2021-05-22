CDF       
      obs    8   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�t�j~��      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�:�   max       P�v(      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �C�   max       >O�      �  l   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>���
=q   max       @E�\(�     �   L   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��\(�    max       @vmG�z�     �  )   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @P            p  1�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�\�          �  2<   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��j   max       >y�#      �  3   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�-]   max       B,�K      �  3�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B,�K      �  4�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?Y�K   max       C�G�      �  5�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?P��   max       C�B�      �  6�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7|   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min          
   max          =      �  8\   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min          
   max          )      �  9<   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�:�   max       P��      �  :   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�L�_��   max       ?�E8�4֢      �  :�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �C�   max       >�P      �  ;�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�=p��
   max       @E�\(�     �  <�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?׮z�H    max       @vmG�z�     �  E|   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @P            p  N<   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @���          �  N�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         E
   max         E
      �  O�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?!-w1��   max       ?�C��$�     �  Pl               9   	               U      "            7   &      
      s      ^      
                  C            D            &      :         �                           �      N��1N�W�OU�NO�{PNlN��pO?@�O���O���N7�-PS��OF�SO���O^��N,e�N�#�O���O[o NY�N(dDNf�cP�v(N�JP;�NKLN�)�N���NB�O=�Na�OH=NO��O(wN'r�N%�	P#jN'�.N0��NR�#O�EVO1O�ҵN�8N�P$M���NX#N��O�ٺO��COg��N�CsM�:�O��WN�y�NRU!�C����D���ě��D���o:�o;D��;D��;ě�;ě�;ě�<t�<#�
<49X<49X<D��<D��<D��<T��<u<�C�<���<�j<���<���<���<�h=+=C�=�P=�P=�P=�w=�w=�w=�w='�=,1=0 �=<j=@�=D��=T��=]/=]/=m�h=u=u=�O�=���=���=�j=�h=��>O�Z[^^gt������tgb[ZZZZ������������������������������������������������������'!&)5B[t���tg[NB5'��������������������#/<HUVY]USH</+#)5BN[t��weQNB)0,6=BO[fkllljihaOB80mhmst{����wtmmmmmmmm	#<DHYanxraU<#	�������������������������������������������#-/<CHE<5/#
��dbgtx}xutrkgdddddddd

�� )/;HORTSG;/"��������������������`[Y[afntsqnjba``````��������������������������������������������		5dgd[N5���������������������������)6@DDA5)�����������������������qnlt{�����������xtq����������������������������������������	#/<HLSSLH</#
	�����������������������������������������������������������������$#
�	
$$$$$$$$$������������������������������������������ ����������������������������������������������������������}zv����������������������������
/<?DB:/#
������������������������������������������������6IOYD3)�����c]htwyttthcccccccccc��������������������sruz����������{zssss�������')#����������(..) �����LHLNV[gt�������tgXNLtst}�����������}yttta_aanntqnaaaaaaaaaaa���������

�����ZUUYamuz||zpmaZZZZZZ���������������������n�q�zÇÌËÇÇ�z�n�a�a�W�W�a�l�n�n�n�n�ݿ�������������ݿԿҿݿݿݿݿݿ�������������������������������������������*�6�M�U�W�O�M�:�$���������������àìù�����
���������������åÛÚà�����������������������������������	��"�/�6�5�2�,�$�"��	��������������	�.�;�G�T�`�c�a�]�X�O�M�G�;�.�"����$�.�����¾ʾϾ־վʾ���������s�p�s�x������Ź������������������ŹŶŹŹŹŹŹŹŹŹ��5�Z�����������s�N��������ݿ��������(�F�N�R�P�N�A�5�(�������������������������ŹŭŠŗŔōőŠŹ�����뾘���������¾ʾоɾ����������������������������������������������������������������(�4�<�A�I�A�4�(����
�������/�;�H�T�a�m�z�����~�m�a�H�;�"�����/àæùþ��������ùìàÓ�}�z�n�zÇÎÓàE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E������������������������������������m�y��~�y�s�m�`�T�L�T�V�`�i�m�m�m�m�m�m�������
�$�0�I�S�B�����ƾƪƟƏƌƞƦ���f�s��������������������s�m�f�c�c�f�f�U�b�i�|Ł�~�x�k�U�0��
������������#�U���������������������������������������俸�Ŀѿؿѿ˿Ŀ��������������������������;�H�N�T�Z�Y�T�H�;�5�/�/�/�1�;�;�;�;�;�;ÇÓàìñìàÚÓÇ�z�w�z�|ÇÇÇÇÇÇ������	������������������������"�.�2�.�*�"��
�	�	��	���"�"�"�"�"�"�M�Z�f�j�h�m�|�����s�f�Z�M�A�;�5�9�A�M�ܻ������л������a�l�n���������ûлܻ��!�-�2�F�S�\�S�F�:�-�!������������������žʾ׾׾޾׾ʾ����������������������������������������������������������'�@�L�Y�b�j�g�Y�'����ܹϹù������ùߺ'���ʼӼӼμʼ����������������������������/�:�<�>�<�8�/�&�#�"�#�#�/�/�/�/�/�/�/�/�)�6�9�6�0�5�)����� �)�)�)�)�)�)�)�)��	���(�4�A�M�Z�k�q�o�f�Y�M�A�4��
������	��"�.�;�B�?�8�.�"��	��������`�m�y����������y�`�T�;�*�$�)�)�.�;�S�`�/�5�9�0�/�#�#��#�$�/�/�/�/�/�/�/�/�/�/¿��������������¿²°¦¡¦ª²¹¿¿�����ʼ���ռʼ����������������������������������������������������������������	���������
������	�	�	�	�	�	�	�	�5�A�N�X�Z�S�N�H�A�7�5�0�(�(�(�(�5�5�5�5�5�?�N�`�b�]�N�A�5�(���	�����$�0�5�t�g�Y�G�A�B�E�B�N�g���������������������������������������˻:�F�K�S�X�S�O�F�C�:�6�-�!���!�-�7�:�:����������������������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�ŔŠŭŹſ��ŹųŭŠŜŔŒŏŔŔŔŔŔŔE+E7ECEPE\EUEPECE7E3E*E+E+E+E+E+E+E+E+E+  8 n ( I [ 1 W 3 D a Q E Y g L " 5 b 6 T = : : G � E e G e c Q V ^ E \ ; ] N 1 l  2 S K R E 0 R  $ 1 <   Q  �  �    �  �  �  �  �  '  b  %  �  �  �  Y  �    �  �  G  y  �    O  F  7  �  p  �  Y  �  <  �  d  O  n  F  f  z  �  h  �  <  �  �    -    1  j  �  �    �  �  ���9X��j%   <ě�=Y�;ě�<T��<�<�h<49X=�Q�<��=,1<�9X<�t�<�C�=�+=L��<�C�<�j<�t�>$�=+=�h<�=C�<�`B=#�
=�o=t�=u=��=}�=49X='�=�"�=<j=ix�=T��=��
=e`B=���=]/=�7L>Q�=u=��=��=�1=ě�=\=���=���>y�#>I�>�wB	�\B��B��B?
B�#B�B�?BơB�B	�7BSB�xB3�B��B	�B$��A�-]BCBo�B �B,�KB�B �cB�HBB
��B��B"IBwvB!��B�9B�CBoB$�,BX�B5MB#DmB 8,BG�B�&BB�(B��B�B+'B�@B@'B _yBgqBO�B	t<B�EB�aB��A�'B\kB	�$B��B��B�kB��B��B�sB�QB��B	�VB<�B@7BAB��B	��B$��A��B@B�B��B,�KB,|B �oB�B;�B
?ZB�+B"?VB�WB!��BEDB��B@B$��BE�B�LB#@HB ��B��B��B~@B�bB��B��B hB�WB>>B :~BߚB�rB	EfB�.B�zB�ZA�q�BJ�A��A�1A���A�pA�Q�A�_�A���Acf�AKn�A��A�Y�A��MA�-�AK�hA��A6�MA�S[A��]C�G�?Y�KAj+B0XAD�8A랸A�:�AvO�A�)�AʈUA�v�A^(A?�&@���@l�KAPqA���?�]�@��A�A�o�A9�A\�(Ah�RA�EA�.�@���@�I�A���A�mA��A��jA�3@{& @��NC��A�cC���Aǥ/A! A�yA���A��A���A���Ab�AKPA�e1A���A�T+A��iAJ��A��,A74�A��pA�|�C�B�?P��Ai��B��AD��A�qBA���AvA�	;Aʁ<A�x6A][�A=(.@��r@d!APugA��?�[�@�I�A��A։�A9�A\
NAh�WA�cA��@���@��A�x�A��A�}	A�B1A�R�@|�@���C��iA��C���               9   	               U      "            7   '            s      ^      
                   C            E            &      :         �                           �                     '         !         5      #            !               =      /                        '            /                   !         '            !   !         
                                                                           )                              %                                                   !   !         
         N��1N�W�N�t�O�C�N��fN? �O?@�O]�VO1�N7�-O���O*��O?y�O^��N,e�N�#�O�T9O�=NY�N(dDNf�cP��N̡�O�(!NKLN�)�N���NB�O6�Na�O1��O�`OxN'r�N%�	O��$N'�.N0��NR�#O!�>O1Oj��N�8N��O�ɘM���NX#N��O�ٺO��COg��N�CsM�:�O(��N�y�NRU!  �    �  �  Z     �  �  A  _  �  �  �  z  �  Y  	  �  �  �  �  	c  �  	�  _  y  �  �  �  �    
  �    �  �    %    x  W  U  �  U  Z    H  �  z  T    .  �    )  �C����o%   <�h��o:�o<49X<t�;ě�=8Q�<o<��
<#�
<49X<49X<�/<��
<D��<T��<u=y�#<��
=m�h<���<���<���<�h=�w=C�=�w=#�
=��=�w=�w=�%=�w='�=,1=u=<j=�+=D��=]/=�9X=]/=m�h=u=u=�O�=���=���=�j>�P=��>O�Z[^^gt������tgb[ZZZZ������������������������������������������������������.+,/5:BNQXTPNIB5....��������������������#/<HUVY]USH</+#)5BH[`gqig[NB5'@:86:BGO[`ghihf\ZOB@mhmst{����wtmmmmmmmm#/4<HLMMLHF</#�������������������������������������������#-/<CHE<5/#
��dbgtx}xutrkgdddddddd

	
"/;BGJKKF?/"	��������������������`[Y[afntsqnjba``````��������������������������������������������)5CMRPI=���������������������������)355/)������������������������qnlt{�����������xtq���������������������������������������� #/6<HIPOHF</*#  ������������������������������������������������� ����������������$#
�	
$$$$$$$$$��������������������������������������������� ���������������������������������������������������������������������������������������
#/6<<><6/#
�����������������������������������������������)7?A>4)���c]htwyttthcccccccccc��������������������sruz����������{zssss�������')#����������(..) �����LHLNV[gt�������tgXNLtst}�����������}yttta_aanntqnaaaaaaaaaaa��������

������ZUUYamuz||zpmaZZZZZZ���������������������n�q�zÇÌËÇÇ�z�n�a�a�W�W�a�l�n�n�n�n�ݿ�������������ݿԿҿݿݿݿݿݿ������������������������������������������*�6�C�K�I�C�6�+������������������*ù������������������������ùîïùùùù�����������������������������������������	��"�/�6�5�2�,�$�"��	��������������	�;�G�Q�T�[�Z�W�T�H�G�D�;�.�-�"���"�.�;�����������ƾʾоϾʾ�����������������Ź������������������ŹŶŹŹŹŹŹŹŹŹ�(�A�N�i�n�g�a�N�A�5�(�����������(����(�5�D�N�P�O�N�A�5�(����
�������������������������ŹŭŠřŞŠŭŹ�ƾ����������¾ʾоɾ����������������������������������������������������������������(�4�<�A�I�A�4�(����
�������;�H�T�a�f�o�t�s�m�a�T�H�;�/�"����/�;àìù��������þùìàÓÊÇÃÇÐÓØàE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E������������������������������������m�y��~�y�s�m�`�T�L�T�V�`�i�m�m�m�m�m�m�����������'�#����������ƩƧƩƴ���̾f�s�����������������s�o�f�d�d�f�f�f�f�<�I�U�Z�c�b�Z�L�0�#��
�� ��	��#�0�<���������������������������������������俸�Ŀѿؿѿ˿Ŀ��������������������������;�H�N�T�Z�Y�T�H�;�5�/�/�/�1�;�;�;�;�;�;ÇÓàìñìàÚÓÇ�z�w�z�|ÇÇÇÇÇÇ��������������������������������"�.�2�.�*�"��
�	�	��	���"�"�"�"�"�"�M�Z�b�f�i�g�k�z����s�f�Z�M�A�<�6�;�A�M���ûлܻ�������̻������g�l�p�����������!�-�1�F�F�S�V�F�:�-�!�������������������žʾ׾׾޾׾ʾ�����������������������������������������������������������'�3�@�L�S�V�Q�L�@�3������ֹչ������ʼӼӼμʼ����������������������������/�:�<�>�<�8�/�&�#�"�#�#�/�/�/�/�/�/�/�/�)�6�9�6�0�5�)����� �)�)�)�)�)�)�)�)�4�A�M�X�[�Z�S�M�E�A�4�(�&����#�(�3�4�����	��"�.�;�B�?�8�.�"��	��������m�y���������y�n�m�`�T�G�:�3�;�A�G�T�`�m�/�5�9�0�/�#�#��#�$�/�/�/�/�/�/�/�/�/�/¿����������¿²¦ ¦¬²»¿¿¿¿¿¿�������ͼּۼڼмʼ��������������������������������������������������������������	���������
������	�	�	�	�	�	�	�	�5�A�N�X�Z�S�N�H�A�7�5�0�(�(�(�(�5�5�5�5�5�?�N�`�b�]�N�A�5�(���	�����$�0�5�t�g�Y�G�A�B�E�B�N�g���������������������������������������˻:�F�K�S�X�S�O�F�C�:�6�-�!���!�-�7�:�:����������������������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�ŔŠŭŹſ��ŹųŭŠŜŔŒŏŔŔŔŔŔŔE+E7ECEPE\EUEPECE7E3E*E+E+E+E+E+E+E+E+E+  8 L . : @ 1 P & D D J $ Y g L  " b 6 T ; 9 ( G � E e 1 e c P R ^ E H ; ] N " l  2 V B R E 0 R  $ 1 <   Q  �  �     /    X  �  �  |  b  9  �  �  �  Y  �  7  8  �  G  y  �  �  o  F  7  �  p  '  Y  �     �  d  O  �  F  f  z  Y  h  �  <  �  �    -    1  j  �  �    d  �  �  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  E
  �  �  �  �  �  �  s  a  M  6    �  �  �  p  <    �  o   �          �  �  �  �  �  �  �  �  �  �  ~  v  m  f  _  Y  �  �  �  �  �  �  �  �  �  �  w  ]  +  �  �  �  U  *      �  3  [  s  �  �  z  k  [  G  7  %  "    �  �  �  '  �  �  2  o  �  �  �  �  �  �      #  F  Y  9     �  T  �    u  �  �  �  �                  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  k  X  E  3  #      �  �  �  �  �  �  �  �  �  �  �  �  �  �  ^  .  �  �  j    �  �  8  �  �  �  !  6  ?  A  9  ,      �  �  �  �  P  
  �  9  �  p  _  W  O  G  =  -      �  �  �  �  �  �  h  M  #   �   �   �  �  G  �  �  �    8  U  �  �  n  5  �  �  9  �  Y  �  �  �  �  �  �  �  w  ]  B  &    �  �  �  �  L    �  �  4  �  �  m  }  �  �  �  �  �  �  �  �  �  �  �  n  5  �  �  �  3  �  z  u  o  i  m  n  a  Q  ?  "    �  �  �  d  X  V  8    �  �  �  �  �  �  �  �  �  �  �    n  _  O  @  2  $      �  Y  J  :  +      �  �  �  �  �  �  �  }  h  R  >  0  !      _  �  �  �  	  	  �  �  �  4  �  v    s  �  �  5  v  �  &  n  �  �  �  �  �  �  �  {  [  "  �    V  �  �  �  {   L  �  �  �  �  �  �  {  l  \  G  3      �  �  �  ~  S  )   �  �  �  �  �  �  �  x  m  _  N  ;  '    �  �  �  �  p    �  �  �  �  �  �  �  }  v  m  e  ]  T  L  ?  '     �   �   �   �  �  4  m  �  �  	  	I  	b  	[  	B  	#  	  �  }    y  �  �  f  �  �  �  �  �  �  �    p  ^  J  5    �  �  �  W    �  B   �  w  �  	P  	�  	�  	�  	�  	�  	�  	�  	�  	�  	&  �  #  �  �    �  l  _  P  B  4  )  /  4  9  6  '    	  �  �  �  �  �  �  �  v  y  c  M  6       �  �  �  �  w  k  t  x  q  b  6  	  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  p  c  W  J  =  1  �  �  �  �  �  �  �  �  �  �  �  {  r  m  h  ^  U  L  B  9  .  ~  �  �  �  �  �  w  U  +    �  �  Y  �  z  �  3  �    �  �  �  �  �  �  z  s  l  e  [  M  ?  1  $       �   �   �          �  �  �  �  {  M     �  �  I    	  �  �  
  g  
  
  
  	�  	�  	a  	0  	  �  �  z  A     �  E  �  �  �  �    �  �  �  �  �    Y  )  �  �  x  6  �  �  u  /  �  �  �  Z    O  �  �  �  �  �       
            !  b  �    S  �  �  �  �  |  u  o  i  c  \  T  H  <  1  %        �  �  �    E  p  �  �  �  �  �  �  �  P    �  C  �  �  �  �  W      �  �  �  �  �  �  �  �  �  �  �  |  l  [  J  8  $    %  �  �  �  �  K    �  �  �  k  G     �  �  �  w  I    �             �  �  �  �  �  �  �  �  �    "  ^  �  �  �  p  _  V  U  M  O  ]  i  u  v  g  I    �  �  ?  �  f  �  &  W  E  2        �  �  �  �  �  �  v  c  P  <  %    �  �  �  	    %  5  M  T  O  3    �  �  -  �  <  �  
  `  �  ~  �  �  �  x  k  _  R  A  .      �  �  �  �  �  �  �  n  \  G  N  S  P  C  4    �  �  �  �  _  .  �  �  �  t  3  �  Y  ~  p  �  *  U  T  ;    �  �  [  �  U  s  ~  ~  U  
�  �  W        �  �  �  �  �  �  �  �  �  �  t  O  *    �  �  �  H  '    �  �  �  �  b  0    �  �  �  v  N  %  �  �  �  }  �  �  �  �  w  b  M  6      �  �  �  y  J    �  �  M  �  z  v  n  [  D  *  
  �  �  �  f  /  �  �  k    �  S  �    T  <  =  ;  1  "  
  �  �  m  7  Y  ;    �  �  /  �  �  ]      �  �  �  �  �  b  C  #    �  �  �  [  +  �  �  �  �  .  #        �  �  �  �  S    �  �  u  F  $    �  �  �  �  �  �  �  �  s  [  L  <  .         �  �  �  �  �  m  L  �  �  �        �  �  �  >  �  k  �    �  L  t  (  �  �  )    �  �  �  t  J    �  �    I    �  �  ]  &    �  �    
�  
z  
9  	�  	�  	q  	1  	  �  b    �  c    �  �  i  �  �