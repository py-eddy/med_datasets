CDF       
      obs    9   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?pbM���   max       ?�dZ�1      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�gj   max       P���      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �t�   max       >%�T      �  t   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @@         max       @F(�\     �   X   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�G�z�    max       @vz�Q�     �  )@   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @          max       @O�           t  2(   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ȵ        max       @��@          �  2�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �e`B   max       >��+      �  3�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       B�8   max       B/�1      �  4d   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       B�3   max       B/�n      �  5H   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?-�!   max       C���      �  6,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?1�R   max       C���      �  7   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          E      �  8�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9      �  9�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�gj   max       P���      �  :�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���
=p�   max       ?�Z���ݘ      �  ;�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��/   max       >%�T      �  <h   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @@\(�   max       @F(�\     �  =L   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�G�z�    max       @vw
=p��     �  F4   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @          max       @O�           t  O   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ȵ        max       @��@          �  O�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >y   max         >y      �  Pt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���E��   max       ?�U�=�L     �  QX      &   U                                       -   :                  X      >         .   <      (   D   
   �                   '   W                           	      �      	      )   Or
O�.O�}�M�gjN0�N�OFФN%KNd��N��fN�N�I-O�N�KN��P
��O�fhN���O'��N�A�N���N���Pkn�O|X*P���N<�N�3�PuQO��8OaV�O��UP���N��)PJG'O�+�Oq��O��O/XUO��O_56O�A�N(nN�*|Nx�Om�wO~��O4r`N�E�N,�N�u,N�VO��N�v8NK�UN��O�^>Né�t���1��C���C���o�e`B�t���`B��`B��`B�ě��D��%   ;ě�<o<t�<#�
<#�
<e`B<e`B<u<�C�<���<�1<�9X<�9X<�9X<�9X<�/<�<�<��=o=+=C�=t�=t�=�w=�w=�w=D��=P�`=T��=T��=Y�=]/=]/=�+=�O�=��=��P=��
=��=� �=��=ȴ9>%�TWS[gtw����~tgf_\[WWMFEN[gt������tg`[UNMEBBFLUanz�������nUHE�����������������������������������������������������������
#/<HCHNHE</#��������������������"#$///<A</#"==>BBDOQRX[[\[ROIDB=��������������������������������+(&/7<HOTUaga_UH<0/+����
�������HDCDO[hhtutmih_[QOHH���������	����������������������������[agptv~����tkg[[[[[[�����)+)&��&)))���
��������  !(������������������������������������������)5C[t�gZJ>���)0-) 6?BFKN[\gillgg[TNB66����������������������������������������#/<HSUalaUPH</# nvttu{������������zn����)6OXXUPB)������921<<HUY^abaUTH<9999MS[gt����������tg[QM���������������������������������������������
#/10*#	��������������
"!
����MHHMO[dhtx���ytk[ROM[Z\^ht�������|vthg\[�����)5BILKHB5)��A?BCNUXQNBAAAAAAAAAA����� ������������������������������������������������������������$)/1)��*66>CDFDC6*�z|��������������������' ����������� 

�������������

��������������������������������������������))'$)169863/,)).9@B96)������



 ���������zÇÍÓÊÇ�z�y�n�a�W�U�N�S�U�a�n�q�z�z���
���#�!���
����������������������E�E�E�E�FFF F*F/F$E�E�E�E�E�E�E�E�E�Eͺ������������������������n�{ŇňōŇ�{�n�k�n�n�n�n�n�n�n�n�n�n�n�����������������������������������������������������x�i�q�w���������������������������������������"�#�/�1�/�"����	����	�� �"�"�"�"�����(�4�(�!����������ݽҽݽ���²¿������¿µ²¦¤¦®²²²²²²²²��������������������ùùïùý������������������!������ ������������������"�"������������������"�.�;�G�P�T�T�T�T�T�R�G�;�8�.�*�"��"�"�������������������������������������������������������ƳƭƬƮƭƴƾ���h�u�vƁƃƁ�u�r�h�\�W�X�\�_�h�h�h�h�h�h�������������������������������y�y�������/�;�H�T�W�V�T�H�<�;�:�2�/�.�/�/�/�/�/�/�����������������������������������������h�u�|ƁƈƎƚƟƚƎƁ�z�u�h�c�\�[�\�d�h�����о��Z���������Z�A���齒���������������������������������{�����������׿���@�g���������������f�M�(����ӿؿ�����������������������������������������������$�)�&�$�������������������������"�Y�r������f�4����˻ƻϻܻ���N�[�g�t�t�g�N�5�0�/�4�>�N���!�&�%�!�������������������(�5�A�Z�g�k�q�o�b�Z�N�5�(������H�[�_�Y�]�[�L�/�������������������	�"�H�m�z���������������z�z�y�n�m�k�k�m�m�m�m�B�[�j�u�y�z�l�[�O�B�6�.�(�� �������B�4�M�Z�f�����������������f�Z�M�A�(�)�4���Ŀѿݿ��ݿѿĿ����y�s�}�������������`�m�p�q�y����{�m�`�T�F�;�.�'�.�;�G�T�`���
��#�/�<�=�B�@�6�/�#���
�����������'�4�6�E�M�V�V�M�@�4�/�'�"����������ʼּ�����ּ���������������������)�5�B�I�s�z�y�r�g�[�N�B�5�"������U�b�l�n�x�n�b�U�P�Q�U�U�U�U�U�U�U�U�U�U�/�;�A�>�;�.�+�"������"�/�/�/�/�/�/�<�H�I�P�J�H�<�/�#��#�.�/�7�<�<�<�<�<�<������������
���������¿½¿��������ĚĦĳ���������������������ĿĳĠĕėĚ�#�0�<�E�I�N�O�L�I�C�<�0�+�#�!�����#�m�y���������������}�y�m�`�_�X�Y�`�a�m�m���ûлڻһлû������������������������������������������������������������������{ǈǔǡǬǭǳǭǥǡǔǈǇ�{�v�r�{�{�{�{D�D�D�D�D�D�D�D�D�D�D�D�D�D�DtDkDjDlDxD��f�r���������������������r�p�g�f�e�f�f���������������y�y�m�y�z�����������������@�=�@�L�N�Y�e�r�~���������~�r�j�e�Y�L�@�:�F�S�n�x�x�l�e�_�S�F�:�!�����!�-�:EuExE�E�E�E�E�E�E�EuEtEiEfEaEiEmEuEuEuEu D W 4 D ( D ! D d n {  ' O V D + G T * : i  c N ^ 1 h * B ; @ - 4 T e @ - Q G G B  P B Q 3 $ $ A D  K / Z 9 2    &  K  �   �  B  I  �  I  �  &  1  �  S  �    �    �  �  �  �  �     ^  �  l    P  �  �  V  .  �  �  �  Q  k  �  m  �  V  D  �  �  �  :  z    D  �  �  �  *  `      ؼ#�
<�o=�C��e`B�T���49X;��
�D����o:�o�D��<49X<���<49X<u=Y�=�C�<u<�t�<�t�<��
<��
=�"�=0 �=�1<���=+=�7L=� �=m�h=�\)=���=,1>fff=aG�=D��=�7L=�+=q��=��->
=q=e`B=�%=u=��
=��=�t�=�{=���=��T=�{>��+=��`=��=���>O�>>v�B	��B	��Bk�B��B�B9@BsB��BaB��B �NB�/B�uBYnB	{B�B�8B	�VBioBBJLB��B"c�B�BfBBR�B�xB!U�B�qB�#B:�B��BL�B
i�B��B��B��Bu�B�^B[8BN�B LBB�/Bb-B�B�,B/�1BɓB@B�B�qB�B,OkB6IB�1B�B	��B	"B@WB��B�BCB3UB��B��B�kB DBB�B�aBFB.�BB�B�3B	�B�BʒB��BÕB"��B��B@�B��B�|B!��B�*B��B@�B��B>$B
DvBǄBG�B7�BNBB0BEQB@�B?�B �B<�B��B�#B[�B/�nB�oB��B¶B�|B?�B,A�B?CB�
B?�A�Z�A�ٞC���@R?�A��?-�!A�͘A�b5A���A0�PA�H�A�nA�;hA�[pAc?OA���B>�BPAs<A�4A�c�B��A3��A���A��B�GB�@ȋAA���A�ϚA���A�*�A�A��AB*Au�tAhY-A�~�@�9�@��A��uA��A`�A�.�A��A�3	A��Al�6@��AK�9B/�C���@�:A�D?�@�VDC���A�}oA���C���@Pd,A�?1�RA���A��+A���A.��A�5�A΃AҐeA�l�Ac�A���BK B_Asf�A�y6A�M�BdA18�A��dA�nKB�'B�#@��yA��:A��kA���A�iaA�y�A׍AD�eAvE�Ai oA���@��@��6A�ypA�A`��A�A��WA�{A��Al�h@�P�AK�LB�C��@�[�A?�V@|q�C���      &   V                     	                  -   ;                  Y      ?         .   <      )   E      �                   '   X         	                  	      �      	      )            #                                       -   !                  9      E         1            9      -   #                  #                                                                                                                                    '            9         !                                                                  NZ�mO?�OI�M�gjN0�N�O��N%KNd��N��nN�N�I-N�F�N�KN��*Ov�YOSz?N���O'��N�A�N���NJY�O�[�Oi�O��TN<�N���O�ԛOBWEOaV�Oi�P���N��)O���O��[Oq��O��TO��N�BO=��O�iN(nN�*|Nx�Om�wOs�UO�LN�E�N,�N�u,N�VOIDN�v8NK�UN��O��Né  O  I  =  �  �  K  �    �  k  E  �  �  w  �  �  $  S  �  �  �  �  o  �  4  ^  J  a  �    �  7  �  �  m  m  �    �  �  �  �  l  �  +    �    ;  �    �    �  �  L  ׼�/�u<T����C���o�e`B�ě���`B��`B�ě��ě��D��;�o;ě�<t�<���=+<#�
<e`B<e`B<u<�t�=�%<�9X=q��<�9X<���=+=8Q�<�=�P=#�
=o=���=\)=t�=�P=,1=,1=0 �=�\)=P�`=T��=T��=Y�=aG�=ix�=�+=�O�=��=��P>�-=��=� �=��=���>%�Tbggtz����thgbbbbbbbbXQNP[gtz�������tge[XRKJJOUanz{||}~zynaUR�����������������������������������������������������������#/<??HHH></)#��������������������"#$///<A</#">>?ABOPPVZZOFB>>>>>>��������������������������������.,+/<HOUU\UNH<3/....����
�������FEFO[hjhh^[OFFFFFFFF���������������������������������������[agptv~����tkg[[[[[[�����)+)&��&)))���
��������	!�������������������������������������������)5=BIJF8/��)0-) BDKN[fgjig[NBBBBBBBB����������������������������������������#/<HSUalaUPH</# zyz}�������������}}z����6OUWRB)������921<<HUY^abaUTH<9999c_^`gt�����������tic�����������������������������������������������
#/10*#����������
!
����JIOOZ[hitu}���vtg[OJd^\\]`hnt�������ythd���)5=BCDA;3)�A?BCNUXQNBAAAAAAAAAA����� �������������������������������������������������������������')+)���*66>CDFDC6*�z|��������������������' ����������� 

�����������	

	����������������������������������������������))'$)169863/,))-9@B=62)������



 ���������n�z�zÃ�z�o�n�a�_�V�a�b�n�n�n�n�n�n�n�n�����
�������
��������������������E�E�E�E�E�F
FFFFE�E�E�E�E�E�E�E�E�Eͺ������������������������n�{ŇňōŇ�{�n�k�n�n�n�n�n�n�n�n�n�n�n�����������������������������������������������������~�o�s�w�}�������������������������������������"�#�/�1�/�"����	����	�� �"�"�"�"������"��������ݽ�������������²¿������¿µ²¦¤¦®²²²²²²²²��������������������ùùïùý����������������������������������������������"�"������������������.�;�G�N�R�M�G�;�:�.�,�"�.�.�.�.�.�.�.�.��������������������������������������������������������������������ƿ�������h�u�vƁƃƁ�u�r�h�\�W�X�\�_�h�h�h�h�h�h�������������������������������y�y�������/�;�H�T�W�V�T�H�<�;�:�2�/�.�/�/�/�/�/�/�����������������������������������������h�u�uƁƎƒƎƁ�u�r�h�f�_�g�h�h�h�h�h�h�����A�E�M�M�A�,��������ֽ׽ݽ��������������������������������������������(�5�L�Z�^�g�i�g�d�Z�N�A�(����������������������������������������������������!��������������������������������@�M�Y�h�t�f�M�4������ػ߻��[�g�i�t�y�t�r�g�[�N�B�:�9�?�B�N�Q�[���!�&�%�!�����������������(�5�A�N�Z�c�g�k�i�[�N�5�-�(���
���(�/�J�R�U�U�R�T�C�/����������������	�"�/�m�z���������������z�z�y�n�m�k�k�m�m�m�m�)�6�B�O�\�c�e�c�[�P�B�6�)�������)�M�Z�f����������������f�Z�M�A�4�)�,�4�M���Ŀѿݿ��ݿѿĿ����y�s�}�������������G�T�`�m�o�p�y�~���z�m�`�T�G�?�;�.�(�.�G�
��#�/�;�<�@�>�<�4�/�#���
�
�	�	�
�
��'�3�4�@�B�M�M�O�M�@�9�4�2�'�%�������������ʼּ߼�����ּʼ������������)�5�B�N�]�i�o�n�g�_�[�N�B�5�)�����)�U�b�l�n�x�n�b�U�P�Q�U�U�U�U�U�U�U�U�U�U�/�;�A�>�;�.�+�"������"�/�/�/�/�/�/�<�H�I�P�J�H�<�/�#��#�.�/�7�<�<�<�<�<�<������������
���������¿½¿��������ĚĦĳĿ������������������ĿĳġĚĖĘĚ�#�0�<�A�I�L�M�I�H�<�5�0�.�%�#�����#�m�y���������������}�y�m�`�_�X�Y�`�a�m�m���ûлڻһлû������������������������������������������������������������������{ǈǔǡǬǭǳǭǥǡǔǈǇ�{�v�r�{�{�{�{D�D�D�D�D�D�D�D�D�D�D�D�D�D|DzD{D�D�D�D��f�r���������������������r�p�g�f�e�f�f���������������y�y�m�y�z�����������������@�=�@�L�N�Y�e�r�~���������~�r�j�e�Y�L�@�:�F�S�l�w�l�d�_�S�F�:�-������!�-�:EuExE�E�E�E�E�E�E�EuEtEiEfEaEiEmEuEuEuEu 8 M & D ( D ' D d f {  # O 9 $  G T * : f L - E ^ . g  B 9 A -  R e C ' R : < B  P B R ) $ $ A D  K / Z 4 2    h  �  �   �  B  I  :  I  �  �  1  �  �  �  �  �  �  �  �  �  �  }  9  �  �  l  �  
  �  �  �  R  �  5  �  Q  N  9  *  �  <  D  �  �  �    =    D  �  �  �  *  `       �  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  >y  �  �  �        2  G  .  �  �  ~  )  �  I  �  G  �  &  �    .  B  H  H  F  ;  3  *    �  �  �  S  �  �    �  �    	�  
(  
_  
�  
�    4  =  8    
�  
k  
  	�  	-  �  �  �  �  "  �  �  �  �  v  i  ]  N  >  .      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    |  y  v  s  p  m  j  g  d  a  _  \  K  C  <  4  ,  %            �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  o  U  5       �  �  �  �  �            
      �  �  �  �  �  �  �  �  �  �       �  }  v  p  i  b  \  X  V  U  S  Q  O  L  F  A  ;  6  0  +  @  R  d  f  ]  V  T  S  W  \  _  b  f  o  w  �  �  �  p  X  E  K  R  Y  _  f  m  p  r  t  u  w  y  }  �  �  �  �  �  �  �  9  C  2  $      �  �  �  �  �  �  �  �  r  `  H  *    p  �  �  �  �  �  �  �  o  H    �  �  �  U    �  �  �  o  w  n  d  [  Q  E  9  -         �  �  �  �  �  �  �  q  \  �  �  �  �  �  �  �  �  r  _  M  9  $    �  �  �  �  d  -  �    /  F  _  u  �  �  �  }  g  G    �  �  N  �  S  u   �  �    Y  �  �  �    "       �  �  �  "  �    c  �  �  �  S  N  J  E  @  :  4  /  )  "      �  �  �  c  K  :  (    �  �  �  �  �  �  �  �  �  �  ~  s  g  [  O  H  D  @  ;  7  �  �  �  �  �  �  �  �  �  }  u  o  i  b  \  U  O  H  B  ;  �  �  �  �  �  �  �  |  q  c  V  I  :  )      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    z  w  x  y  z  {  |  }  A  �  �    E  Q  <  9  (  n  k  U    �     d  �  �  �  �  $  �  �  �  �  �  �  w  a  F  $  �  �  �  v  9  �  �  *  v  �  �         �  �  �    $  )  1  +    �  h  �  A  L  T  ^  Z  V  S  O  K  G  B  =  7  2  -  (  #              <  C  I  I  I  J  J  I  B  9  )    �  �  �  Y    �  �  �  %  ,  5  O  ^  ^  M  C  K  <  (    �  �  R    �  �  �  6  s  V  z  �  �  �  �  �  �  ~  M    �  n    �  �  ^  C  �    ~  {  u  m  d  W  E  .    �  �  �  \    �  ]  �  K  �  �  �  �  �  �  �  �  �  U  %  �  �  �  E  �  u  �  6  [  ~    '  4  /    �  �  �  k  ,  �  B  �  �  V  )  �  m  `   �  �  �  �  �  �  �  �  �  t  c  S  E  6  (    �  �  �  �  x    4  c  P    �  *  �  �  �  �  _  �  �  �  $  \  
�  �  k  _  l  d  T  B  .      )  &      �  �  �  e    �  o    m  ]  M  =  +      �  �  �  �  q  _  M  7    �  �  j    �  �    [  $  �  �  H  o  b  I  )    �  �  Z    �  R  �  �  �      �  �  �  �  X  '  �  �  v  "  �  Y  �  P  �  _  �  �  �  �  �  �  �  k  S  7    �  �  �  x  E    �  �  "  q  �  �  �  �  v  E    �  �  T    �  k    �  �  d  �  �  �  E  �  �  �  �  �  �  D  �  �  /  
�  
>  	�  �  �  �  �  �  �  �  �  �  x  o  f  \  R  E  9  ,        �  �  �  �  �  l  Y  F  1      �  �  �  �  s  P  -  	  �  �  �  t  :  �  �  �  w  c  O  <  1  %      �  �  �  �  �  �  x  �  �  E  +    �  �  �  �  s  R  )  �  �    >  �  �  =  �  ;  �    �    �  �  �  �  �  j  G  #  �  �  P  �  �  I  �  }  2    �  �  �  �  �  �  �  �  �  �  j  D    �  �  \    �  Z  �    �  �  �  �  r  U  8    �  �  �  m  3  �  �  �  {  8  �  ;  *    	  �  �  �  �  �  t  S  2    �  �  �  �  a  9    �  �  �  �  �  �  �  r  W  <       �  �  �  �  o  P  0      �  �  �  �  t  d  V  ?  '    �  �  �  �  n  F    �  �  `  �  e  �    T  |  |  ]  -  �  r  �  �    $  
  r  �  �      �  �  �  �  �  �  k  K    �  �  J  �  �    �  
  T  �  }  q  c  U  G  7  '    �  �  �  �  �  ~  d  J  4  $    �  �  �  �  �  �  �  �  �  �  v  L    �  �  �  l  G    �  @  A  *    �  �  �  w  K    �  �  N  �  �  4  �  R  �    �  �  }  H    �  �  V    �  �  %  �  D  �  *  �  �  2  �