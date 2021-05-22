CDF       
      obs    6   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�Q��R      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       Mòd   max       P�<|      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �T��   max       >o      �  \   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>Ǯz�H   max       @E�G�z�     p   4   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?���P    max       @vq�����     p  (�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @P�           l  1   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�^        max       @�           �  1�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �H�9   max       >cS�      �  2X   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B,�#      �  30   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�o]   max       B,��      �  4   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�O�   max       C�^>      �  4�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��   max       C�Y      �  5�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  6�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9      �  7h   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          5      �  8@   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       Mòd   max       P��@      �  9   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���p:�   max       ?��t�k      �  9�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �T��   max       >+      �  :�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>���R   max       @E�G�z�     p  ;�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?���P    max       @vq�����     p  D   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @P�           l  L�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�^        max       @�{�          �  L�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         E�   max         E�      �  M�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�bM���   max       ?�F�]c�     �  N�                                                            !   s      �   L      %   �   �         $            /      5         "   &               !   '      s         W      PNB�eNË�O�N܍IN ǇN�C�O�5OG=tO (�O��XO�O�KN^~	N�S3O���Nީ�O���N�NLO��P�<|O�?Pr��P��N\�O'��PI�PZlOV(N��AO>=�MòdN��dO�ONcN;��O��}N���O���O�;�O�ӚN#�hOg�O.rN ��O�xO�.�N���O��#Np�4N��]O��Om6O�f�T����o�e`B�D���#�
�o���
�D����o��o;o<t�<t�<#�
<49X<D��<D��<�o<��
<�j<�j<�/<�/<�/<�`B<��=o=+=\)=�P=�P=8Q�=<j=H�9=H�9=P�`=P�`=]/=q��=y�#=�o=�7L=�C�=�\)=�t�=��P=��P=���=��=���=�Q�=��=��>o���������������������}�����������������#*/;<@BC<5/)#dedhhkrtv������zthdd��������������������./39<HPUX[UH</......���������������������)6BOXOJDB6)	�!#&/1<@HMSTMH</,#�����'%�����"/9;HOHC;7/"	������������������������������������������������������������������������#/<HOTNH</$$# &*0<UbcfcXTI<0*$ ���������������������()/)����������������������������������%,EH?5�����^agz������������zmb^���������������������)6D`c^JB5%��������������������������������������������������#
�������% $+N[gt�������gNB-%����������������������������������������KJKOY[bhjpy|tnlh[TOK�������������������������

	�����������������

����������
#'/0.(#
��������������������������������������������������������������������$&&�����������������������)6IQVVSOB6)�#./1/#



"#####��������������������������������������������������������)#�����)<>:1+,)�����)+-.( ��������������������������������
�����������������]XZabmqzypma]]]]]]]]������������������������������������

������������n�{ńŅń�{�n�l�e�m�n�n�n�n�n�n�n�n�n�nE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��(�5�=�A�L�N�P�A�@�5�(���������(�.�;�G�S�T�`�m�o�m�g�`�T�G�;�3�.�*�)�.�.�#�/�6�<�=�<�6�1�/�-�*�&�#� �#�#�#�#�#�#�z�{ÇÓÕÚÓËÇ�z�u�p�n�y�z�z�z�z�z�z�(�5�A�N�Z�^�_�\�Z�O�N�K�A�5�3�-�+�(�&�(���ʼؼּټּּۼּѼʼ�����������������EEEEE*E*E,E*E%EEED�D�D�D�D�E EE���������	�����������������������������T�a�m�r�w�x�z�|�|�z�v�a�X�T�L�F�E�H�Q�T������������� ������������������������������������������ù÷ñùù��������������ÇÓÖàéëàÛÓÎÍÇ�ÆÇÇÇÇÇÇ�a�m�z���������z�m�a�T�H�;�5�*�"�#�/�H�a�����������������������������������������������������r�e�[�^�f�r��4�A�M�Z�f�l�i�f�`�Z�M�A�4�2�-�/�4�4�4�4�f�h�j�j�f�f�Z�T�P�U�Z�c�f�f�f�f�f�f�f�f�tāčĚĤĪĴĸĲĦĚčā�t�e�e�c�e�o�t�
�0�I�vŇřŞŕ�{�U�#��������ļĵ�����
�����������������������������s�p�z������Çàé����ÿìÓ�n�R�G�B�I�[�Q�U�a�n�|Ç�����	�/�T�c�f�`�T�;��	������������������*�+�6�/�*��������������ù������ùìàÖÓÇÀ�z�x�y�zÇÓàìù��!�C�V�`�a�\�L�-����ݺɺ������ֺ����)�6�_�g�i�e�Q�6�'�����������������������'�2�@�4�'�����ܻػܻݻ���6�;�B�O�X�[�`�[�O�I�B�?�=�6�)���)�/�6��'�4�?�M�S�Y�g�Y�M�@�4���������	��f�s�����w�s�f�e�f�f�f�f�f�f�f�f�f�f�f�;�H�O�T�L�H�A�;�/�/�,�/�0�2�;�;�;�;�;�;���	��"�%�.�9�9�.�"��	�����۾������`�m�y���������������y�m�`�T�I�C�G�H�T�`���Ľнݽ��ݽнĽ��������������������������(�4�A�G�M�K�A�6������������ùϹܹ������ܹܹܹ׹Ϲù����ùùùÿ������ �&�*�'� �����ҿɿѿڿ���[�g�t¤�t�[�N�B�5�,�8�D�N�[���׾��������׾ʾ�������������������ǭǡǝǔǒǑǔǜǔǔǔǡǢǥǭǭǭǭǭǭ�����������������������������������������l�x�������ܻ�ܻлû����������x�p�h�g�l�����������������������������������������>�3�,�3�0�7�L�e�s�����������~�r�e�Y�L�>Ƴ����������������������ƚƁ�xƁƇƔƧƳ�/�<�H�S�U�X�U�M�H�<�4�/�#��#�)�/�/�/�/D{D�D�D�D�D�D�D�D�D�D�D�D�D�D{DiDgDgDoD{�������������������������������������ŭŹ������������Źŭũũŭŭŭŭŭŭŭŭ��'�4�D�K�H�=�.���лŻʻлػ���������ûȻĻû����������������{����������E�E�E�E�E�E�E�E�E�EuEpEmEoEuExE�E�E�E�E� 0 R ( @ � 8 @ ^   ! W H � D - U , % � 9 E ; 1 / R V '   < a H ^ P e  O   Q  R M s * � X , F -  G & Q ^   \  �  :    �  �  F  �      �  G    �  S  4  ,    �  e  �  K  �  8  �  �  U  �  �  �  �    �  Y  �  k  _  �  >  L  4  x  .    d  �  �  �    �  �  6  E  �H�9�o%   ��`B�ě���o;�`B<���<e`B<u<�t�<�/<�/<��
<ě�<�h=��=��<���=e`B>I�=D��>O�;=��=t�=�7L>H�9>cS�=m�h=D��=��=D��=T��=��=ě�=e`B=��=�+=���=��=��`=�hs=���=���=���=��=�`B=�->H�9=�E�=��>?|�>%>T��BFpB?*B�BwB�B	�B�BB�B��A���B��B��Bp�Bq�B��B&nNB��BդB��B��B ɽB��BSLB�iB"D�B"\rB	�B!��B��Bw�B�ZB�lBc�B)B��B"%BBʧBT{Bu�B 'B�B,�#B,�B2ABQOB��B�&B�pBzA��Bz.B^ BY�BG{B?�B?�B��B�yB=�B�;B�GB�gBD�A�o]B�B-BG�B�B�[B&@WB�BXFB@�BDB �PB��B�B�xB"A*B"A3B	2RB!��B�rBAoB�RB�&B@)B=-B�PB"=�B�oBBzBF'B�XB�;B,��B,~�BG�B�VB��B��B�iB�A�~3BIBH�B�=A�2zC�^>A�� Af��A���A���A��h@��C�dA�K�A�swA�A��Aʃ�A���A�X@�c�A<��A@^AޘaA���A�*�A�`�A���A�ȟA���@[��A��k@� A�R�@ʓ�AC>�A���A[��AkcA(p�A4��>�O�A�	A�AP��B{A!��@��)ArP�?�-B�qAÝfC�ϪA�CA��@���@��XC��A�l�C�YA�@�AgD�A�\cA���A�y?@���C�g�A��>A��#A�	�A΂?A�n�A���A�{v@��A<�A>��A�JA�wQA���A�{
A�s�A��A�X�@[ezA��g@��A�c�@�ACQ�A���A\��AkCA'
�A5��>��A�L�A���AQAB��A"i�@��pAr�:?���B��A�}�C�׬A���A�z%@��@��kC�U                                                            "   s      �   M   	   %   �   �         %            0      5         "   '               !   '      s         W      Q                                                               9   '   3   7         3   /                                       %               #                  %                                                                     5         1            #                                       !               #                  !      NB�eNË�O�N�N ǇN�C�O�5O&O (�O*�N�u�Nt�N^~	N�S3O���Nk�}O8VaN�NLO��3P��@O̸�O}uPU��N\�O��O��qO�5�OH��Nd�.N�ɋMòdN��dO�ONcN;��O���N���O�h�Ond�O�",N#�hOg�O.rN ��O�xO�ŜN���O,Np�4N��]O���N��OwS�  j  E    �  �  �      f  �  7  �  �     X  �  �  �  L  U  
�  s      �  �  !  �  �  T        �  	�  �  �  �  D  �  @  �  �  �  �  �  1  z  �  +  E    d  ��T����o�e`B�t��#�
�o���
��o��o;��
;ě�<�o<t�<#�
<49X<�C�<��
<�o<��
<ě�=t�<�==�P<�`B=t�=�;d=�v�=t�=�w=L��=8Q�=<j=H�9=H�9=P�`=ix�=]/=u=�o=�+=�7L=�C�=�\)=�t�=��P=��-=���=��=���=�Q�=�G�=��`>+���������������������}�����������������#*/;<@BC<5/)#oort�����toooooooooo��������������������./39<HPUX[UH</......���������������������)BGEB@6)"�!#&/1<@HMSTMH</,#�������	����
"/;?>;1/"������������������������������������������������������������������������+)//<GHOIH</+++++++++'$'05<IUX^]YUKI<60+���������������������()/)����������������������������������!';CD5)���facmz������������zmf������������������������)9?Z^YIB5!���������������������������������������������������������������3029N[gt������tg[E:3����������������������������������������NOPY[hmsqkh`[XQONNNN�������������������������

	�����������������

����������
#'/0.(#
��������������������������������������������������������������������$%%�����������������������6GOTUUROB6)#./1/#



"#####��������������������������������������������������������)#�����)<>:1+,)�����)+,*'%!��������������������������������������������������]XZabmqzypma]]]]]]]]�������������������������� �������������
	��������n�{ńŅń�{�n�l�e�m�n�n�n�n�n�n�n�n�n�nE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��(�5�=�A�L�N�P�A�@�5�(���������(�G�T�`�c�c�`�T�G�C�A�G�G�G�G�G�G�G�G�G�G�#�/�6�<�=�<�6�1�/�-�*�&�#� �#�#�#�#�#�#�z�{ÇÓÕÚÓËÇ�z�u�p�n�y�z�z�z�z�z�z�(�5�A�N�Z�^�_�\�Z�O�N�K�A�5�3�-�+�(�&�(���ǼʼּּӼ׼Ҽʼ���������������������EEEEE*E*E,E*E%EEED�D�D�D�D�E EE�����������������������������������������T�a�m�n�s�s�r�m�a�^�T�R�K�J�T�T�T�T�T�T��������������������������������������������������������ù÷ñùù��������������ÇÓÖàéëàÛÓÎÍÇ�ÆÇÇÇÇÇÇ�a�m�z���������z�m�a�T�H�;�5�*�"�#�/�H�a�������	����������������������������r��������������������������r�o�d�f�h�r�4�A�M�Z�f�l�i�f�`�Z�M�A�4�2�-�/�4�4�4�4�f�h�j�j�f�f�Z�T�P�U�Z�c�f�f�f�f�f�f�f�f�tāčĚģĩĳķĳįĦčā�t�e�e�d�f�p�t���
�0�I�nŐŕň�{�U�0��������������������������������������������x�t�u�z�~�����n�zÇÓàãììèàÓÇ�z�k�a�`�_�`�j�n���	�;�T�]�a�Z�H�;��	��������������������*�+�6�/�*��������������Óàìù��������ùðìàÓÇ�}�|ÇÓÓÓ������1�;�=�:�-�!�������׺Ѻֺ�����)�6�P�Y�Z�U�B�6�)���������������������'�1�?�4�.�'������ݻ޻���6�8�B�B�O�[�]�[�O�E�B�A�>�6�)�!�)�2�6�6�'�/�4�@�D�E�@�4�'����	���!�'�'�'�'�f�s�����w�s�f�e�f�f�f�f�f�f�f�f�f�f�f�;�H�O�T�L�H�A�;�/�/�,�/�0�2�;�;�;�;�;�;���	��"�%�.�9�9�.�"��	�����۾������`�m�y���������������y�m�`�T�I�C�G�H�T�`���Ľнݽ��ݽнĽ������������������������(�4�A�D�J�G�A�4�(������������ùϹܹ������ܹܹܹ׹Ϲù����ùùùÿ�������%�)�&�����ԿͿѿԿۿ���[�g�t�t�g�[�N�B�5�0�4�:�G�N�[�����׾��������׾;�����������������ǭǡǝǔǒǑǔǜǔǔǔǡǢǥǭǭǭǭǭǭ�����������������������������������������l�x�������ܻ�ܻлû����������x�p�h�g�l�����������������������������������������>�3�,�3�0�7�L�e�s�����������~�r�e�Y�L�>Ƴ��������������������ƳƧƚƎƂƌƗƧƳ�/�<�H�S�U�X�U�M�H�<�4�/�#��#�)�/�/�/�/D�D�D�D�D�D�D�D�D�D�D�D�D{DyDwD{D�D�D�D��������������������������������������ŭŹ������������Źŭũũŭŭŭŭŭŭŭŭ����'�4�?�F�D�9�*����ܻһлջ��������ûƻĻû����������������|����������E�E�E�E�E�E�E�E�E�E�E�E�E�EvEqEnEqEuE|E� 0 R ( 2 � 8 @ S   . L : � D - 0 . % � 9 C -  + R ] "  @ z - ^ P e  O  Q  N J s * � X , ; -  G & E \    \  �  :  (  �  �  F  �    ,    �    �  S    �    �  [  <  �  �  |  �    (    �  �  �    �  Y  �  k    �  '  �  2  x  .    d  �  4  �  I  �  �  �  7  �  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  j  m  q  u  x  |  �  {  r  i  `  W  N  B  0       �   �   �  E  (    �  �  �  �  �  �  �  �  m  m  U  3    �  �  m      �  �  �  �  �  �  �  �  �  k  I    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  g  V  E  5      �  �  �  �  }  a  E  )  �  �  �  �  �  �  �  �  �  {  k  Z  H  6       �  �  �  }        �  �  �  �  �  �  �  �  �  �  �  ~  d  F     �  �  �  �    �  �  �  �  h  H  ,          &  H  k  �  �    f  Q  5    �  �  �  j  <    �  �  u  9  �  �  ^  4    �  Y  d  p  }  �  �  �  �  �  �  �  �  k  M  ,    �  �  n  .  �    &  1  6  7  7  4  *      �  �  �  {  E    �  g  �  <  I  _    �  �  �  �  �  �  �  �  �  �  �  r  ^  I  2    �  �  �  �  �  �  �  �  r  U  5    �  �  �  d  $  �    �       �  �  �  �  �  �  �  o  a  X  [  d  m  t  }  �  �  �  X  M  @  0    	  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  i  H    �  �  W    �  C  �  1  b  z  �  �  �  �  �  �  �  i  F    �  �    �  3  �  j  �  �  �  �  �  �  �  �  y  d  E    �  �  h    �  �  �   �  L  X  e  q  o  k  f  ^  S  I  6       �  �  �  d  &  �  �  P  T  E  &  �  �  �  [  1     �  �  :  �  t    �    ~  �  
  
�  
�  
�  
�  
|  
E  
  	�  	x  	&  �  �  �  F  �  �  �  �  8  _  _  j  n  Z  A  "  �  �  �    S  !  �  �  Z  (  �  p   �  �  
�  �    
  �  K  �  �      �  �  U  �  l  �  +  �  �  �        	  �  �  �  �  �  n  ,  �  ~    �    o  �    �  �  �  �  �  �  �  �  �  }  j  X  F  6  &    �  �  �  _  Q  �  �  �  �  �  �  t  N    �  �  5  �  <  �  �  )  Q  �  
�  Y  �  i  �  `  �  �         �  x    O  j  
V  �  �  0  
  7    �  1  u  �  {  N     �    Y  �  �  J  �  
�  �  {  �  �  �  �  �  ~  v  o  a  I     �  �  �  �  �  �  �  �  �  �    =  M  �  �  
                  #  5  M  g  �  i  �  �  �  �          �  �  �  q  0  �  �  �  �  "   �    
    �  �  �  �  �  �  �  �  x  b  G     �  �  �  �  `    p  b  T  F  9  +      �  �  �  �  �  �  �  �  �  �  �  �  �  {  p  \  ?  "    	  �  �  �  �  �  y  _  I  4  Q  =  	�  	�  	�  	�  	�  	�  	r  	K  	  �    (  �    �  �  o     �  P  �  �  �  �  v  h  Z  L  ;  $    �  �  �  �  Z  D  4  #    �  �  �  �  �  �  �  �  j  1  �  �  n    �  ;  �  �  �  �  �  p  K  -  N  i  M  0    �  �  �  u  M  )    �  �  c  (  =  D  <  +    �  �  �  �  O    �  u  2    �  �  �  H  �  �  �  �  �  �  �  �  o  O  )  �  �  �  H    �  c    $  ~  3  @  6  #    �  �  �  s  :  �  �  r  *  �  �  B  �  >  +  �  }  t  k  b  [  [  \  \  \  [  X  T  Q  N  @  /      �  �  �  �  �  �  �  s  `  H  +    �  �  �  �  X  #  �  �  R  �  �  �  �  y  d  Q  =  #    �  �  �  �  r  J    �  �  s  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  n  -  
  �  �    r  s  u  s  c  A    �  �  8  �  �    $  .  /  +  "    �  �  �  m  /  �  �  I  �  w  �  �  �   �  z  _  E  /  
  �  �  �  �  `  -  �  �    Q  $  �  �  T   �  �  d  �  C  �  �  �  �  �  P     �  /  �  �  c  �  
  �  �  +  %          �  �  �  �  �  �  �  �  �  �  �  �  �  |  E  1       �  �  y  U  /    �  �  �  Y  ?    �  �  :  �  �        �  �  �  �  t  I  
  
�  
X  	�  	/  Z  m  ]  �  �  c  b  S  2     �  �  S    �  �  [    �  h    �     �  c  �  �  �  _    �  �  �  K  �  �  "  �    
l  	|  q  @  �  