CDF       
      obs    :   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�������      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N��   max       P�7�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��h   max       =��      �  |   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�33333   max       @E�z�G�     	   d   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��
=p�   max       @v���
=p     	  )t   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @P`           t  2�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @Ͱ        max       @��           �  2�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �ě�   max       >}�      �  3�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��B   max       B-�c      �  4�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��y   max       B-��      �  5�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?�f�   max       C���      �  6�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?��	   max       C��      �  7�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  8h   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          M      �  9P   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          5      �  :8   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N��   max       P�35      �  ;    speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��*0U3   max       ?�	ԕ*�      �  <   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��`B   max       >&�y      �  <�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�33333   max       @E�z�G�     	  =�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��
=p�   max       @v���
=p     	  F�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @#         max       @P`           t  O�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @Ͱ        max       @�`          �  Pl   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A{   max         A{      �  QT   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�qu�!�S   max       ?�1&�x�        R<            H               	                  
               
   �         R                     	   S   9      ;      "   2   D   	      (      	      *         #   .   Y          `         
   O	�N#{�O9&LP�7�OT˭NZ�N��O�<[O=�N�<qN	G�N� BOf��O��kO=�WN�C�O��O��N���ND�uP��KN� ON��sP#�xN��O)N��hOh�NDpN��Nu�P	�LP��}NB�O��JOT��N�+�P&	]O��O$ߘOQڜO��uN�`{N�U�O%RPmlN���N�'�O��SP�P2fPNJ��OJ��Pf�O xN�%N��LNL����h��`B�ě����
%   ;ě�<o<t�<#�
<#�
<49X<e`B<�o<�C�<�t�<�t�<���<���<�1<�1<�j<�j<�j<ě�<�`B<�h=o=+=+=\)=\)=t�=t�=�P=��=��=�w=�w=�w=�w=�w=�w=�w=,1=,1=49X=49X=<j=@�=P�`=T��=�o=�7L=�hs=���=��=� �=��c]]_gt��������tgcccc��������������������).5BNZY[f`[NB5)�����)Ufg^C�������Zacgwz����������ztaZhhot�������tohhhhhhh��������������������������
#'*#
���� 
))-45@5)& BCNR[git{{tkg[UNBBBB������������������������������mlmooz�����������ztm����������������������������!!��316?BCNOTUY[[VOBB863'2<DHLNGKLKC7/#�����������BHKN[gjigg[NBBBBBBBB��������������������")5B[��������t[*,/2<HJUUVUPH<2/****���������� �������������
/354/
����HFHU_abaUHHHHHHHHHHH�����������������������������������������~���������������������������������������{|~����������������������������������������)6<<:5)������)5EKX[\VH5)��""//34/"���������

��&*6BOQUWVMQOB@6)��������������������������
#/2@D2+#��������#/6>@A<3/����������� 

������
#(/2651/#
��JIN_htvs�������thYQJ������������������������������������������������������������)5M[t}������g[B5) �������������������
!(���������������������������������),-)	������������� ���������
#&#
��������""+/;>HINOOKH;/"*��������)330)*�����!��������������������������������������������#-/8/,#!�U�a�n�z�~Ã��z�u�n�a�]�U�P�K�Q�U�U�U�U�����������������������������������������G�T�]�`�d�e�g�`�T�Q�G�;�7�3�/�2�7�;�<�G�ѿ���5�X�[�S�(���ѿ����������ſ������Z�]�g�s���������������s�g�Z�N�I�L�N�Y�Z�`�a�m�p�p�m�b�`�\�T�O�N�T�`�`�`�`�`�`�`�����������������������������������������<�H�a�v�x�xÁÅ�z�a�H�3�/�#�����#�<������������������������������������������������������������������������������������������������������������������������Óàìùû������ûùìàÓÏÇÁÇËÓÓ�tāĄčĚĢģĦħĥĚĖčā�t�h�c�a�h�t���������������������s�f�Z�U�V�_�c�m����	��"�.�;�A�G�N�N�G�:�.�"�	�������������������������������s�r�s�s�����������5�A�R�Z�d�Z�A�5�(�����������������������ʾԾʾʾ������������������������ĽнԽڽ׽нĽ�����������������������������������������������������������������6�[āďĘęĕ�t�h�E�6�5�)�������������� �����������������������������뻞�����ûлػ׻лû����������}�����������)�5�N�g�g�[�B���� � ���)àìööìãàßÜÝàààààààààà����(�5�A�L�A�@�8�5�(���������������������ֺԺֺغ���������(�5�A�N�Y�Z�_�b�f�Z�N�A�5�*�(�"� �!�&�(�s�������������s�p�i�s�s�s�s�s�s�s�s�s�s���������������������u�s�r�o�s�����������4�A�M�U�Q�M�A�4�,�+�4�4�4�4�4�4�4�4�4�4�
��0�I�Q�Z�`�\�I�0�
�����������������
Ƨ������D�H�F�=�$�����ƳƎ�}�x�w�}ƖƧ�a�m�z�z�z�z�m�a�X�V�a�a�a�a�a�a�a�a�a�aEuE�E�E�E�E�E�E�E�E�E�E�E�E�EuEiEcEaEiEu�-�5�:�:�2�/�/�!�������ݺغ�����-DoD{D�D�D�D�D�D�D�D�D�D�D�D�D�D{DoDjDhDo���������/�T�k�o�j�W�H�"�	������������D�D�D�D�EEE*E;E?E7EED�D�D�D�D�D�D�DӾӾ׾ھ������"�.�5�*�"�	�����;Ⱦӿm�y���������|�y�s�m�`�T�G�D�9�;�F�T�`�m�'�4�@�Y�t�z�r�Y�G�4��
���������'������
�������������������������4�8�A�D�B�A�4�/�(����	����(�,�4�4�����"�(�-�(��	��������������1�5�H�F�9�/�"�	��������������������������������������������������������������f�s�y�u�s�s�r�j�f�Z�R�U�X�Z�]�f�f�f�f�fÓàù��������������ùàÓÑÐÒÒÌÈÓ�����
��"�%� �������²�u²¿����@�r�~�����ʼռ߼�ּ�������o�^�U�I�4�@ǮǴǱǭǥǡǘǔǓǔǙǡǭǮǮǮǮǮǮǮ�������
���#�%�&�#���
��������������'���%�3�<�[�~�����������������~�r�Y�'�	���"�(�.�.�"���	����������������	FF$F1F=FJFTFSFJF=F4F1F-F$FFFFFFF�_�l�x��������������x�v�l�_�]�_�_�_�_�_�6�C�O�Q�O�M�C�>�6�.�*�(�*�6�6�6�6�6�6�6 0 ( H + 2 A y : E L ^ 7 # G 6 b ` 6 > t ( ! l  F 4 K I 2 l  + C 1 L O k P G w ; P q / 0 E P � 6 m \ [  J + @ L K  1  B  �  I  �  �  �  �  H  
  =  �  �  �  �  �  �     �  �  �  �  A  �  0  U  �  �  S  �  |  i  p  e  \  �  H  "  �  �  �     �  �  G  �  �    �    o  �  �    b  �  �  g�t��ě�;ě�=�+<u<t�<e`B=\)<�t�<�/<T��<���=�P=�w<�`B<�j=�w=o<�j<��>}�=�w=\)=�
=<��='�=<j=}�=�P=�w=49X=��=�j=49X=ě�=q��=�t�=�E�=�"�=D��=�+=���=<j=L��=Y�=�{=H�9=T��=��=ě�>\)=�\)=ȴ9>)��=ȴ9=���=ě�=��B	�HB*��B��B6�B �B�bB�)Bz�BA�B�.B�B!��B ��BoB�B��B��B{B�B �B�B��B#YBj B��B�XB�B��BB
�B��B��B��A��BB�B��B��BBB?�B��B3�B nB!^�B 	�B"�xB		YB-�cBB@B�B�BWRB�&A���B{B�cB�B,��B��B	��B*��BC�B�EA���B]�B��B}�B=�B	�B�B!�B �BĞB�YBKB�B@�B��B @�B��B$	B#=>BC�B��B�LB�B:�B�B
�BB�iBʎBB7A��yB��B?.B�/B�eB�sB�LB@AB¾B!��B �B"��B	A,B-��B�/BBrB�B��B8,A�}SB��B�B:gB,��B��A� �@���Ae�A���A��Ah��A��AĎ�A�0A���At)yA��sA��AE>�A_y%AG�mA��AK��A&��@�JIA�GnA��u@�!A�`�A�-AA���@JWIA���A�p�A�óA:�4A�]�B�4A�:C�~@^�C���A�U�C�U'AYi�Ain�@�)�@��A7 EA2nA��A"CIA@�uAͅ�A�u@��)B�A��?�f�A���C���@��B ��A�Z�@��3AeA��-A��@Ah��A�k�AƇ�A��A��As9�A�z*A݉fAF��A_�AIZ>A�u�AJ�	A'��@��Aق�A�}�@�A�{�A�w�A�!�@K�jA��A��=A�MA;�A�r�B�0A���C�R@[�QC��A�CC�L�AY0�Ai�@�@�c�A7Q�A1&A���A!�A@�[A�_RA���@�	B��A�,?��	A���C��@��dB v]            H               	                                    �         R                     
   S   9      ;      "   3   E   
      )      	      *         $   .   Y          `                        M            !                           !            7         '                        %   3               -   !         )            )            -   /         )                        5                                                                                       )               -   !                                 #   #         )            N��[N#{�O9&LP�35O:�NZ�N��O}�GO=�N���N	G�N� BOV�IO�'wO%�!N�C�O877N���N���ND�uOxf�N|4�NȬO� �N��N� N��hO3oINDpN��Nu�O�P&�NB�O�N��N�}P&	]O�N�O$ߘOQڜO�N�`{N�U�O%ROL�N���N�'�O��SO���O�hNJ��OJ��Pf�OLQN�%N��LNL��  h  k  2  2  J  �  (  �    �  �  _    �  n  �    I  :  Z    �    -  *  S    6  o  H    
c  0  #  
�  O  
  m  7  �  �  0  u  �  K  �  �  P  �    
�  �  l  G  1  �  �  ʼ�/��`B�ě�<#�
;�o;ě�<o<�o<#�
<T��<49X<e`B<�C�<�1<���<�t�<�j<�9X<�1<�1>&�y<�`B<ě�=u<�`B<��=o=��=+=\)=\)=m�h=T��=�P=m�h=8Q�=49X=�w=<j=�w=�w=@�=�w=,1=,1=��=49X=<j=@�=q��=���=�o=�7L=�hs=��
=��=� �=��e__cgt��������tgeeee��������������������).5BNZY[f`[NB5)������)5RYXSB5����ihkmuz��������zmiiiihhot�������tohhhhhhh���������������������������

��� 
))-45@5)& GINZ[\gtwxtg[NGGGGGG������������������������������unnqrz������������zu��������������������������� ��316?BCNOTUY[[VOBB863
#.<@DFF>2/#
�����	�������BHKN[gjigg[NBBBBBBBB��������������������:889=BN[gltvusjg[NB:-/07<CHQOHC<7/------����������  ������������
  
����HFHU_abaUHHHHHHHHHHH���������������������������������������������������������������������������������{|~�����������������������������������������)3552-)����)5CNRRNB5)��""//34/"������

����&&)6BBOORPOFB<62*)&&��������������������������
#/2@D2+#������� #/3<>?:/
����������� 

������
#(/2651/#
��NMO[htu������toh[XRN������������������������������������������������������������EELN[gqtz��tg[TNEEEE�������������������
!(����������������������������������%))&������������������������
#&#
��������""+/;>HINOOKH;/"*��������)330)*����!��������������������������������������������#-/8/,#!�U�a�n�z�|Á�}�z�r�n�a�^�U�R�L�S�U�U�U�U�����������������������������������������G�T�]�`�d�e�g�`�T�Q�G�;�7�3�/�2�7�;�<�G�ݿ���1�B�I�H�9�*����ݿȿ��������Ŀ��Z�g�s�w�����������s�g�Z�Y�N�P�X�Z�Z�Z�Z�`�a�m�p�p�m�b�`�\�T�O�N�T�`�`�`�`�`�`�`�����������������������������������������#�<�H�a�n�r�t�y�u�n�a�H�<�/�#���� �#������������������������������������������������������������������������������������������������������������������������Óàìùû������ûùìàÓÏÇÁÇËÓÓ�h�tāčĚĠġĦĤĚĔčā�x�t�h�d�c�b�h���������������������s�f�^�^�c�d�h�s��	��"�.�;�?�G�L�L�G�6�.�"���	������	����������������������s�r�s�s������������#�-�5�A�5�(������������������������Ⱦľ������������������������������ĽнԽڽ׽нĽ����������������������������������������������������������������6�B�O�[�h�q�v�t�l�h�[�O�B�6�)�'�%�&�*�6�����������������������������������뻞�����ûλлӻлû����������������������5�B�N�[�e�k�i�g�b�[�N�B�5�)�����)�5àìööìãàßÜÝàààààààààà����(�5�>�<�5�5�(����������������������ֺԺֺغ���������5�A�N�T�Z�\�_�_�Z�N�A�5�/�(�%�#�%�(�.�5�s�������������s�p�i�s�s�s�s�s�s�s�s�s�s���������������������u�s�r�o�s�����������4�A�M�U�Q�M�A�4�,�+�4�4�4�4�4�4�4�4�4�4���
��#�/�>�J�K�B�0�#�
��������������������������#�#������ƳƘƌƈƊƐƜƧ���a�m�z�z�z�z�m�a�X�V�a�a�a�a�a�a�a�a�a�aE�E�E�E�E�E�E�E�E�E�E�E�E�EuErElEuEuE�E���!�*�)�!�!������������������DoD{D�D�D�D�D�D�D�D�D�D�D�D�D�D{DoDmDnDo���������/�T�k�o�j�W�H�"�	������������D�D�D�D�EEE*E7E;E7EED�D�D�D�D�D�D�D߾Ӿ׾ھ������"�.�5�*�"�	�����;Ⱦӿm�y���������|�y�s�m�`�T�G�D�9�;�F�T�`�m�'�4�8�K�U�O�M�B�4����������	���'������
�������������������������4�8�A�D�B�A�4�/�(����	����(�,�4�4�����"�(�-�(��	��������������"�/�0�4�3�/�-�"��	������	����������������������������������������������f�s�y�u�s�s�r�j�f�Z�R�U�X�Z�]�f�f�f�f�fÓàù��������������ùàÓÑÐÒÒÌÈÓ�������
�������������¼²¦¿������f��������˼ּӼ���������r�j�d�a�\�\�fǮǴǱǭǥǡǘǔǓǔǙǡǭǮǮǮǮǮǮǮ�������
���#�%�&�#���
��������������'���%�3�<�[�~�����������������~�r�Y�'���"�'�.�-�"���	�����������������	�FF$F1F=FJFTFSFJF=F4F1F-F$FFFFFFF�_�l�x��������������x�v�l�_�]�_�_�_�_�_�6�C�O�Q�O�M�C�>�6�.�*�(�*�6�6�6�6�6�6�6 ' ( H ,  A y < E J ^ 7 ! 4 3 b 6 2 > t  & f  F , K A 2 l  % 0 1 ? > l P E w ; = q / 0 ; P � 6 X E [  J ' @ L K  	  B  �  �  (  �  �    H  �  =  �  �    _  �  �  �  �  �  �  ~    �  0  
  �  �  S  �  |  q  �  e  E  �  "  "  �  �  �    �  �  G  G  �    �  �  %  �  �    A  �  �  g  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  A{  h  h  h  f  a  U  B  )    �  �  �  Y    �  �  :  �  3  e  k  k  k  k  k  k  g  c  `  \  W  Q  K  E  ?  5  *       
  2  2  0  +  $      �  �  �  �  |  E    �  �  v  :   �   �  (  �  �  $  2  *      �  �  �  �  �      �  I  X    �    -  <  D  G  I  I  H  F  >  *    �  �  �  �  ]  0  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  _  C  (     �   �  (  )  *  +  )  '  $        �  �  �  �  �  �  �  x  c  M  }  �  �  �  �  �  �  �  �  u  {  j  7  �  �  H  �  �  3  �      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  W  0    �  �  ~  @  �  >  z  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  _  O  =  *    �  �  �  �  �  �  �  �  `  �  u  6  �  �  h          �  �  �  �  �  m  Q  *  �  �  �  b  4    �  �  d  z  �  �  �  �  �  �  ~  b  F  (    �  �  s  2  �  �  �  h  k  n  j  f  _  W  L  @  2  "    �  �  �  �  �  �  p  T  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  `  D  (    �  �  �  �    �  �  �  �  �  �  �  f  B    �  �  I  �   �  3  :  @  E  H  I  I  D  ;  -      �  �  �  �  �  L  �  5  :  5  1  ,  '  #          
     �   �   �   �   �   �   �   �  Z  H  8  +  !            �  �  �  n  -  �  �  X    �  �  �    y  �  �  "  y  �  �        �  z  �  �  �  	�    s  �  �  �  �  �  �  �  �  �  y  M    �  �  c  ,  !  J                �  �  �  �  g  M  '  �  �  �  M    �  �  (  �  �  C  �  �  �     ,  *    �  �  b    �  �  !    \  *  (  %  "          �  �  �  �  �  �  �  i  J  +     �  R  R  R  S  N  H  ?  4  )      �  �  �  �  �  f  .  �  �          �  �  �  �  �  �  k  8    �  �  -  X  �  M   �    '  1  5  0  #    �  �  �  K    �  x  (  �  [  �  U  �  o  j  e  `  [  V  Q  L  F  A  :  2  *  "      �  �  �  �  H  :  ,        �  �  �  �  �  �  z  e  O  >  .       �      �  �  �  �  �  �  �  �  �  s  a  O  ;    �  >  �   �  	�  	�  
%  
E  
X  
b  
Z  
F  
  	�  	�  	]  		  �  )  �  �  �  �  �  �  �  �      ,  '  	  �  �  �  R    �  >  �  m  �  X  ]  #      �  �  �  �  �  �  }  i  U  ?  )  
  �  �  �  d  =  	Y  	�  
  
X  
�  
�  
�  
�  
�  
|  
2  	�  	T  �  �  +  ?  4    d  �    &  6  A  I  N  N  C  +    �  �  e    �  �  �  �  �  
B  
b  
v  
~  
q  
4  	�  	i  �  w  �  g  �  <  �  �  +  d  �  �  m  h  U  8    �  �  �  �  �  �  d  0  �  x    �  q    �  
�     7  0    
�  
�  
�  
D  	�  	�  	6  �  �  �  �      �    �  �  �  h  M  ,  	  �  �  �  x  U  4    �  �  �  �     �  �  �  �  �  y  g  O  3    �  �  �  t  E    �  M  �  o  �  �  �  �  -  0  +    �  �  �  T  '  
    �  �  ]      !  u  i  ]  Q  D  8  (      �  �  �  �  �  �  �  �  }  l  [  �  �  �  �  �  �  �  �  �  �  �  r  \  @  #    �  �  �  �  K  G  >  +      �  �  �  �  �  p  T  6    �  �  �  g  -  H  �  �  �  �  o  k  q  �  �  �  �  �  �  H    �  �  �  f  �  �  �  �  �  �  �  �    r  f  Z  P  I  B  ;  ;  ;  <  =  P  ?  .      �  �  �  �  �  l  E              �  �  �  �  �  �  �  p  W  F  3    �  �  X    �  M  �  |  �    n  �  �      �  �  �  �  �  _    �  �  -  �  F  �     �  	~  
(  
g  
�  
�  
�  
�  
�  
o  
'  	�  	f  �  y  �      �  �  �  �  �  �  �  �  m  T  >  )    �  �  �  �  �  }  ]  9     �  l  R  @  %    �  �  �  �  b  2  �  �  <  �  _  �  \  �    G  %  �  �    )  !  
  �  �  t    
�  
  	x  �  �  e  h  �  +  1      �  �  �  �  �  h  C    �  �  �  d  /  �  �  �  �  �  �  j  O  4    �  �  �  �  �  �  f  K  -  	  �  �    �  �  �  �  �  }  h  Q  7    �  �  �  �  �  o  \  K  <  .  �  �  �    g  T  A  .    �  �  �  �  `  7    �  �  �  n