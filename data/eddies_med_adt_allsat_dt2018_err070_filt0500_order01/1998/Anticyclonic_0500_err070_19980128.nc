CDF       
      obs    8   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��l�C��      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�M   max       PR�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       >�P      �  l   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��
=q   max       @F��G�{     �   L   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?������    max       @A�=p��     �  )   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @P            p  1�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @���          �  2<   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �t�   max       >["�      �  3   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�Z�   max       B4��      �  3�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B59      �  4�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?|�N   max       B��      �  5�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?�P�   max       B��      �  6�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          n      �  7|   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          1      �  8\   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          )      �  9<   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�M   max       P�v      �  :   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���p:�   max       ?�I�^5@      �  :�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��C�   max       >!��      �  ;�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��
=q   max       @F��G�{     �  <�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?������    max       @A�=p��     �  E|   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @P            p  N<   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @          �  N�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D�   max         D�      �  O�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�=�K]�   max       ?��t�k     �  Pl               	   *            	   "      S   "   m   
         %      V   ]         (                  
            K                  !            2      )      
      	         c      BN�tN|�ZN	/)Nڛ�N���P6�N�jPOG��O)�[N]y[O�
jOm'cPR�OKzP:�wN�OXxN��Od�O7FPBGP?��M�MO7\O��uO797N[SO@��O8�N\ZAN�{�NR&�N�H�N��P9AN+GN�h�O��GNf�N�[�OQ��N�U�O �N��POo�
O�=aN��N��NoٗNw�'O�N0��O͞�N�O^E����D���D��$�  $�  ;D��;D��;ě�<o<e`B<e`B<e`B<���<���<��
<�1<�1<�1<�1<�1<�j<�j<ě�<�<�<��=C�=\)=t�=�P=��=��=0 �=0 �=8Q�=@�=H�9=P�`=e`B=m�h=m�h=q��=u=y�#=}�=�C�=�O�=�O�=���=��
=�{=Ƨ�=ȴ9=�;d>z�>�P��������������������)6BCJB<61)!5..6BCFDB65555555555c_[[cgtz�zvutjgcccc#/<DHKNH<1/,#LR[X[tz��������th[OLMNMO[hmsttutqh[OMMMM��������������������=BO[ht���|th[XQOFEB=����������������������
0UbjldUI0#
���*6?CILJEC96*FEUq����������naUMHF�������������������������
#.46586/#
����))58BFJKEB<5)#?==@BN[]fgklhg[NHB??)-133.)&$)/69BOWXUOIB6)bchlty���������tslhb�������������������������)5=FYUB9��97;;<HNIH;9999999999�
)5;BFJB;5)%���������

�������������������������&$&
#/2<?DC</+#����������������������������������������/+'/1<<HILJH<72/////679:<HJJJHD<66666666���������������������������������������������5BLMF5)�������������������������}z������������}}}}}}����������������������������������������<<<3/#"!#+/<<CHHH@<<����������)6A961)��������������������#0;<EGE<0#��$2EIJNNB5)
" "/CHITZ[ZTQH;/"qnlot�������������tqrqtu|���������vtrrrr������������������������������������������ 
 #$#
�������XUUUZaamz|~zzmhaXX6:<HTSNHC<6666666666rr����������������zr���������������������������������������Ⱥ��������Ǻ�����������������������������ÓØÙÛÕÓÇ�~�z�p�zÁÇÏÓÓÓÓÓÓ���'�*�(�'���
������������6�B�O�[�h�k�l�h�[�O�B�6�)�"�)�+�6�6�6�6������
��������������������M�s�����������s�Z�C�4�����������(�M�����ʾϾѾξʾ��������������������������*�/�6�D�I�C�6�*����������������*���������������������~�q�q�r����������������ʼռмʼ����������������������������������������������������p�a�P�X�u�������T�`�m�������������y�m�`�T�P�G�F�C�C�G�T�(�A�Y�s���������N�A�(�����������(��������������� �����������������������������.�H�H�@�5�)���������������������������������������ŹŭŢŭŮŵŹ���������n�{ŇŔşŝŖŔŇŅ�{�n�b�`�Y�\�b�j�n�n�f�n�r�~�y�r�f�Y�M�@�:�@�F�M�Y�`�f�f�f�f������,�4�9�4�(�������ܻٻۻ����_�l�v�x�|�x�l�a�_�T�R�F�:�3�)�,�-�:�G�_�s�����������������������g�W�N�J�L�T�g�s�����)�B�P�W�V�Q�E�6���������úðù���/�;�H�H�H�B�;�/�-�)�/�/�/�/�/�/�/�/�/�/�f�j�s�t�t�t�t�s�n�f�Z�S�M�J�L�M�Q�Z�d�fŹ��������������������������ŹŰŦŠŰŹ����'�0�3�9�7�3�'����������������5�A�B�N�P�N�A�5�)�.�5�5�5�5�5�5�5�5�5�5�`�m�v�y�{�z�y�r�m�`�T�G�=�;�4�1�;�G�T�`��������������������������������|�}�����	��"�.�5�.�"���	��������	�	�	�	�	�	�	��"�$�"���
�	��������������	�	�	�	���������������������������������������������������s�k�f�c�_�f�s�z������ìù����������ùìàÓÏÓÓààìììì�������������������y�m�`�Q�L�N�_�n�������m�y���������y�m�e�f�m�m�m�m�m�m�m�m�m�m�T�a�m�m�y�z���z�m�a�\�T�Q�O�T�T�T�T�T�T�����������ǽĽ������������y�r�s�r�o�|���׾�������׾־ϾоѾ׾׾׾׾׾׾׾��s�s�t¦¯¬¦�s�s�����!�-�:�F�R�S�G�:�4�-���������������������������������������������������������z�s�f�Z�P�M�H�I�M�Z�f�p�s�������s�}�������������s�f�`�\�[�f�g�s�s�s�s��������"�#��������ƧƎ�y�yƁƐƣƳ���
��#�0�<�?�F�I�K�I�=�0�#���
����
�����������������������������s�j�^�g�y���	�� �"�/�4�/�/�"��	����������	�	�	�	�����������������������������������������������
�	������������������������������ǡǪǭǯǭǪǡǘǔǒǈǉǔǠǡǡǡǡǡǡŇŔŠŭŹ����������ŹŭŠŔœŊŇłŇŇ�����������������������������������������!�_�x����������x�l�_�S�F�.�#�����.�:�<�:�5�.�!������!�%�.�.�.�.�.�.¦²¿��������������¿²¦¡¦ > X 3 T e M A H @ 4 U 2 C < - 5 1 B - @ @ 4 ? _ # % @ % 3 f x v H < X ; X ? r B W @ P & X  T N U @ @ > 0 Z � .  �  �      �  �  �  �  �  r  U  �  �  �  *    >  �  �  g  g  B    g  �  �  ;  �  �  u  �  �  �  
  �  $  �  f  �  �  �  �  �    �  �  �  �  �    �  _  L  :  i  Ǽt���o;o<�o<t�=49X;ě�<�/<�h<�j=D��=C�=��=T��>�<�=<j=�w=m�h=��=�;d=�h<�/=,1=�O�=L��=��=u=e`B=#�
=D��=49X=D��=�hs=�F=H�9=u=���=u=���=�Q�=�7L=�1=��P=�`B=ȴ9=�;d=��T=�1=�Q�=�v�=�l�=���>T��>��>["�B"D�B��B�QB	��Bm�Bf,BynB��B��B"�QB&5�B/��B��B�B��BfB�B�?B�dBy�B�BYA��
B��B��B!c�B�ZB��BִB"�6B�=B�GB��B!�#B�&B ��B �$B+�gB4��BuKB�&B.}B��B%�B�>A�Z�B
�OB
cMBc�Bf�B�A�-yB!-B�)B��B��B"GB�gB��B	�TB�`BC�B��B��BP<B#�B&;�B/�JBA�B��BR�B7�B�vB�B:B?�B�AB?CA���B�B��B!@�BĪB<�B�{B"8�B�MB�GBNtB!��BDpB �tB ��B,�B59B�qB�[B=�BE^B%��B?�A���B
��B
@B?�BC%B��A�~ZB?7B?�B=B9@!ۈA��|?��AلAӢ�A:�ANA��+@���@��@�WAj2�A�[�A�R;A��EA���A�f~@� ,@�q@�ųA��0A��EA�Z�A@�0A�]&?|�NA��Ag�lA��A]7�A�5�A�cACC�A���An�qAl�A��HA�(AS��A���@j_1@��ABj8ACN�BH�A�A���A�OA��A��B��A�=A��h@��mA��A��@$	A���?�P�AكA�H A8�OAN�!A�x@��@��|@��Ah�A��$A���A�vWA���A�_�@�~�@�&�@��VA�~�A���A�l�A?�vA�u�?�k=A��Ah��A��OA]�A��%A�K;AB��A�~�Ao�Al�A�ARNAUdA��z@j�@OABAC�B@\A�~A��[A�~�A���A�B��A�VhA� @���AA�{      	         
   *            
   #      S   "   n   
         &      W   ^         (                              K                  !   	         3      )      
      	         d      C                  -               +      1      +                  -   +                                       +                              )      !                     %                                             )      !                     !                                       !                              '                                 Nkb�N�>N	/)N��N���O�#@N�jPOplN��N, 5OoR\O^�PF�OAY�O�dON�`DN�
^N��Od�N��qO�7�OͧFM�MN�^�N��O797N[SO0�O)�3N\ZAN�{�NR&�NM��N�
�O�WCN+GN�h�Ow��Nf�N�#uOQ��N�U�O �N��P�vOo�
O��N��N��NoٗNw�'O�N0��O�}N�O1@�  G  �    �  i  �  j  ^  �  �  ^  �  �  :  
�  �  �  -    �  
  
�  n    .  �  r  $    �  �  W  �  �  	  �  �  �      �  -  
  �    o  �  �  �  H    5  7  �    ��C��#�
�D��;D��$�  <�o;D��<t�<T��<u<ě�<u=49X<��
=Y�<�9X<ě�<�1<�1<ě�=H�9=aG�<ě�=o=8Q�<��=C�=t�=�P=�P=��=��=49X=D��=u=@�=H�9=aG�=e`B=q��=m�h=q��=u=y�#=�%=�C�=��P=�O�=���=��
=�{=Ƨ�=ȴ9=��m>z�>!����������������������'$)6@BFB765)''''''''5..6BCFDB65555555555b^^ggtv}}xttigbbbbbb#/<DHKNH<1/,#YUUW[cht������thd]YMNMO[hmsttutqh[OMMMM��������������������LJJO[ht~}ztqh[UOLLLL��������������������!#06<IU_^\UNC<0'!�*6<CHKIEC86*�TRZbz����������zna[T��������������������������
&())#
����)5BEIJB5)%A??BFNZ[dgjjgf[NJBAA)-133.)&$)/69BOWXUOIB6)gdehqtv��������tpkhg����������������������
)/5<;3)���97;;<HNIH;9999999999)5BBEB65-)��������������������������������������&$&	#/1<>DC</*#	����������������������������������������/+'/1<<HILJH<72/////679:<HJJJHD<66666666��������������������������������������������)25>ED>5)����������������������}z������������}}}}}}����������������������������������������""#-/<<CH?<4/#""""""����������)6A961)��������������������#0;<EGE<0#��")2DHJNMB5)" "/CHITZ[ZTQH;/"roopt�������������trrqtu|���������vtrrrr������������������������������������������ 
 #$#
�������XUUUZaamz|~zzmhaXX6:<HTSNHC<6666666666�{~��������������������������������������������������������̺������ĺ�������������������������������ÇÓÕÙÓÏÇÂ�z�y�zÅÇÇÇÇÇÇÇÇ���'�*�(�'���
������������B�O�[�h�h�i�h�[�O�B�6�.�6�6�B�B�B�B�B�B������
��������������������(�A�M�Z�e�s�x�r�f�Z�4�(�����
���(�����ʾϾѾξʾ���������������������������*�6�@�C�F�C�?�6�*�&����������������������������������v�v����������������ʼӼμʼ����������������������������r������������������������}�r�f�a�[�f�r�`�m������������y�m�`�T�Q�G�G�D�D�G�T�`��(�A�N�g�|�����s�Z�N�5���������������������������������������������������������)�7�7�1�)����������������������������������ŹŭŰŷŹ�����������������n�{ŇŏŔřŔŔŇŁ�{�n�c�b�[�^�b�m�n�n�f�n�r�~�y�r�f�Y�M�@�:�@�F�M�Y�`�f�f�f�f������,�4�9�4�(�������ܻٻۻ����S�_�l�r�x�z�x�l�_�S�K�F�:�9�/�2�:�F�P�S�������������������������s�g�[�W�V�[�g�����6�;�B�F�F�?�6�)�������������������/�;�H�H�H�B�;�/�-�)�/�/�/�/�/�/�/�/�/�/�f�s�s�s�r�q�q�g�f�Z�W�M�M�M�M�S�Z�[�f�f����������������������ſŹųŵŹ�������Һ���'�0�3�9�7�3�'����������������5�A�B�N�P�N�A�5�)�.�5�5�5�5�5�5�5�5�5�5�`�m�u�y�z�z�y�q�m�`�T�H�>�;�5�8�;�G�T�`����������������������������}�~���������	��"�.�5�.�"���	��������	�	�	�	�	�	�	��"�$�"���
�	��������������	�	�	�	�����������������������������������������f�s�������x�s�q�f�e�b�f�f�f�f�f�f�f�fìù����������ùìàÓÒÓØàæìììì�����������������������y�m�`�Y�T�W�`�}���m�y���������y�m�e�f�m�m�m�m�m�m�m�m�m�m�T�a�m�m�y�z���z�m�a�\�T�Q�O�T�T�T�T�T�T���������������Ľ������������{�t�v�y�z���׾�������׾־ϾоѾ׾׾׾׾׾׾׾�¦¬¦¦�u�����!�-�:�F�R�S�G�:�4�-���������������������������������������������������������z�s�f�Z�P�M�H�I�M�Z�f�p�s�������s�}�������������s�f�`�\�[�f�g�s�s�s�s�������� � ���������ƧƎ�{�zƂƑƤ���
��#�0�<�?�F�I�K�I�=�0�#���
����
�������������������������������o�d�m����	�� �"�/�4�/�/�"��	����������	�	�	�	�����������������������������������������������
�	������������������������������ǡǪǭǯǭǪǡǘǔǒǈǉǔǠǡǡǡǡǡǡŇŔŠŭŹ����������ŹŭŠŔœŊŇłŇŇ����������������������������������������!�-�:�S�_�l�y�������~�x�l�_�S�F�2�(�!�!�.�:�<�:�5�.�!������!�%�.�.�.�.�.�.¦²¿����������������¿²¦¦¦ A B 3 J e = A 4 0 > , 2 K 8 - 9 ) B - H ' 7 ? ^  % @  0 f x v ; < B ; X 7 r 7 W @ P & V  G N U @ @ > 0 I � &  �  @    �  �  �  �  S    I  �  �  �  �  �  �    �  �  #  �  �    #  �  �  ;  r  z  u  �  �  g  �  Q  $  �    �  �  �  �  �    �  �  /  �  �    �  _  L  �  i  q  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  6  ?  F  G  G  E  B  =  6  ,         �  �  �  O    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �         �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  p  [  B  $    �  �  �  �  �  i  \  O  ;  '    �  �  �  �  |  U  '  �  �  �  w  H    �  )  w  �  �  �  �  �  �  �  �  �  �  P    �  �  0  �  r  <  j  g  e  b  `  ]  [  T  K  B  9  0  (      	   �   �   �   �  �  '  W  ]  Y  O  ;    �  �  �  �  U    �  �  '  �  �  M  �  �  �  �  �  �  �  �  x  V  0    �  �  i  !  �  +  �  �  �  �  �  �  �  �          �  �  �  �  �  �  �  �  �  �  g  �  �  -  G  \  \  Q  6    �  �  �  =  �  �  �  @  /  �  �  �  �  �  �  �  �  �  z  V  /    �  �  �  T    �  b   �  �  3  m  �  �  �  �  �  �  �  R    �  I  �    U  >  �  k  ,  3  "    �  �  �  �  [  .  �  �  �  Z    �    �  �  3  �  	�  
  
l  
�  
�  
�  
�  
�  
�  
�  
U  
  	�  	W  �  �  �  �  V  �  �  �  �  t  a  O  <  *      �  �  �  �  v  R  1  $    e  �  �  �  �  �  m  Q  )  �  �  �  e  /  �  �  r  '  �  i  -  $    
  �  �  �  �  �  e  >    �  �  4  �  �  8  o   �    �  �  �  �  �  �  �  v  Y  /  �  �  �  b     �  k  3  \  �  �  �  �  �  �  �    `  @  !    �  �  �  r    {  �    �  	/  	�  	�  	�  	�  	�  	�  	�  	�  	l  	   �  V  �  +  B  !  �  �  �  	f  	�  
   
\  
�  
�  
�  
  
J  
  	�  	W  �  �  �        �  n  h  b  ]  W  Q  K  D  >  7  0  *  #        
    �  �              �  �  �  �  �  �  �  �  �  v  _  F    �  �  �  �  �  �    !  ,  +      �  �  V  �  �    U  p  i  �  �  �  �  �  �  �  �  �  �  v  L    �  �  r  A    �  �  r  i  `  W  N  D  5  &      �  �  �  �  �  �  N     �   �    #      �  �  �  �  T  #  �  �  w  *  �  �  G  2  8  \  |    ~  {  u  i  X  D  -    �  �  �  h    �  ?  �  $   �  �  �  �  �  �  �  �  z  l  ]  O  A  2  &      
      �   �  �  o  U  5  $  P  h  Q  2    �  �  v  C    �  �  h  -   �  W  K  ?  4  '      �  �  �  �  �  o  V  :      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  p  Z  B  +    �  �  �  �  �  �  �  ~  j  Q  /    �  �  �  F  �  a  �  2  �  �  �  �  �  �  �  �  �  �  �  K  �  �  *  �  �  �  �  {  �  �  �  �  |  r  g  ]  R  G  =  3  (      	   �   �   �   �  �  �  �  �  �  �  �    m  Y  D  .    �  �  �  �  �  x  Y  w  �  �  �  �  �  �  k  J    �  �  �  S    �  �  K  �  �          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �          �  �  �  �  �  z  `  C  $    �  �  �  `  �  '  �  �  �  s  U  7    �  �  �  j    �  u    e  �  �  �     -         �  �  �  �  �  �  �  �  f  K  0  &  "  +  L  m  
  �  �    S    �  �  �  o  3  �  �  �  �  �  �  �  ~  X  �  �  �  �  �  r  [  E  .    �  �  �  �  i  C       �  �    
  �  �  �  �  j  9  �  �  Q  �  �  [    �    �    �  o  c  _  X  K  9    �  �  �  .  �  z    �  Z  �  8  h   �  �  �  �  �  �  �  `  2     �  �  W    �  �  =  �      �  �  �  �  �  w  `  D  !  �  �  �  _  )  �  �  �  f  K  0    �  �  �  z  _  F  -    �  �  �  �  x  T  B  .    �  �  �  H  7  $    �  �  �  �  �  c  2  �  �  p  0  �  �  i  %  �    �  �  �  �  �  y  e  Q  =  )    �  �  �  �  �  �  \  1  5    �  �  �  �  e  ;    �  �  �  k  =    �    }    �  7  $    �  �  �  �  �  �  }  i  U  @  -      �  �  �  �  y  �  �  �  �  w  .  �  �  .  �  =  �    ;  
9  	  �  %  �        �  �  �  �  �  �  �  �  n  ^  N  >  .      �  �  �  	        �  �  �  i  
  �  -  �    
w  	�  	  d  �  �