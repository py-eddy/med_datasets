CDF       
      obs    3   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��Q�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N ��   max       Ps��      �  x   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �'�   max       =�Q�      �  D   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��\)   max       @F,�����     �      effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       @ Q��@   max       @vqp��
>     �  (   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @N�           h  0    effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�3�          �  0h   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �C�   max       >Kƨ      �  14   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��.   max       B0��      �  2    latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�Re   max       B0��      �  2�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       @"�   max       C�hr      �  3�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       @�V   max       C�f�      �  4d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  50   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          3      �  5�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          )      �  6�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N ��   max       P-�%      �  7�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��+J   max       ?��/��w      �  8`   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �'�   max       =��      �  9,   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��\)   max       @F,�����     �  9�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       @�Q�    max       @vqG�z�     �  A�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @N�           h  I�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�           �  JP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D?   max         D?      �  K   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?������   max       ?��/��w     �  K�         $                     )         #   -   ,   h            �   �            &         	   2      <   (                        %         *            
   J      	   AN�{�N��O~FNԤ�NO�1OѪ�NZF�NM�{NU2�O���N9�AO��P!>P��P��P/B�Nn�HO��BO\�Ps��O��Nr�Na�O7�5O��bN�HN/a�N��AOOp�!PxqO�v�N��oN50N�ĿN��VN�= N���N��P-�%N ��O�O)��N��RN���N�aTN��@P \O��NP;iO��y�'�P���
��o:�o;ě�;�`B<e`B<�o<�C�<���<�1<�9X<ě�<�/<�<�<��=+=C�=C�=\)=\)=��=#�
='�=0 �=0 �=49X=<j=<j=D��=D��=P�`=]/=]/=]/=aG�=aG�=e`B=ix�=}�=�+=�O�=�\)=�hs=��P=���=�1=�E�=�Q���������������������tsotz����������!!"/<HUadmhaUH</&#!_Z[acnsz���zwnja____#&08:610-#������% �������������������������[[`grtz}ytg[[[[[[[[["&/;C;8/""ACJ[ht�����|sh[OIDA346BOVOOB63333333333�������������^_gr~�������������k^�����
0<IF0#
�����������.5&,)�������������)5=AA=5)���0,05BILHB50000000000khghamz����������zmk���������� ������������Ngtwn[K5)����������
#$!
���������������������������bhmt�����tlhbbbbbbbb2589?FHUalnpnfaUOH<2�������
����������������������������oirt������ztoooooooo����������������������������
#/-(�������������� ��������������)BOV[[YB)���������
#)+*#
��)*+26COSSQOIC6,*))))XT[htvzth[XXXXXXXXXX���("�������������


��������
#$$$$##
�����������������������
 ###
�����ee[it�����������{toe	
#)#
��������������������������
����-)*./<HFF></--------����������������������������������������')*6765)mjdin������������zom������

�����%)-41)�}z|�������������������Ľɽнܽݽ�ݽнĽ��������������������z�v�n�l�h�m�n�zÁÇÍÇÆ�{�z�z�z�z�z�z��������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eͻx�������������x�l�d�l�u�x�x�x�x�x�x�x�x���������%�&�2�5����������žž�����߽��Ľнݽ��ݽнĽ����������������������y�y�������������y�q�n�y�y�y�y�y�y�y�y�y�T�a�b�j�m�q�s�m�e�a�[�U�T�Q�T�T�T�T�T�T�A�Z�f�v�}�~�|�t�f�a�M�4�(�����(�4�A�ʼּټ�߼ּʼʼ��Ƽʼʼʼʼʼʼʼʼʼʿ�"�.�;�G�T�`�i�v�p�d�`�G�.����������A�M�V�f�s����������s�f�Z�5�&�"�$�(�4�A������Ǽɼü���������a�X�\�S�Q�U�_�`��<�U�[�P�<�6�/�������������������
�#�<��<�I�R�W�V�W�P�D�0�����������������
�čĚĦĳıĦĚčĊĊččččččččččŠŹ����������������������ŭŧşŘŔśŠ�����	���&�&�!��	�������޾߾����6�B�N�S�.�&����������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D{DuD{D�D�D�D��a�n�zÇÊÇÀ�z�n�a�Z�W�a�a�a�a�a�a�a�a�����ûǻĻû����������������������������a�m�z�������������������y�m�i�c�a�]�]�a����������������������������������������ÓàìñùþùùìàÓÇ�ÁÇÈÓÓÓÓ�-�:�F�I�J�F�F�:�4�9�-�)�-�-�-�-�-�-�-�-���!�-�3�:�F�F�:�:�-�-�!������������	��/�;�J�T�V�R�H�/�"��	���������������������������������������������������ܻ������	�������ܻлû������ûлܿ"�;�T�\�`�u�x�l�`�G�:�.�"�������	�"�m�q�y�����������y�m�i�`�]�\�`�g�m�m�m�m�ѿݿ���ݿѿ̿ʿϿѿѿѿѿѿѿѿѿѿѾZ�f�j�i�i�f�c�Z�M�H�G�F�M�Q�Z�Z�Z�Z�Z�Z���������	�	����������������������������(�4�A�M�Z�e�f�s�~�y�s�f�Z�M�A�A�4�(�&�(�)�6�B�G�H�B�A�6�*�)�'�(�)�)�)�)�)�)�)�)���&�#�!���������������
���������������������������s�O�D�@�G�Z�i����ĦĳĴĸĵĳĮĦěĢĦĦĦĦĦĦĦĦĦĦ�y���������ĽɽĽ��������������~�y�s�v�yE�E�E�E�E�E�E�E�E�E�E�E�E�EtEiEiElEuE{E��B�N�[�g�j�p�g�[�N�B�>�<�B�B�B�B�B�B�B�B��(�5�A�N�U�N�E�A�5�(����������������������������������������������������������������������������������������������ɺֺۺк̺ͺɺ������������|�|�������6�8�C�M�O�V�\�`�\�[�O�C�6�.�*�%�!�*�3�6�t�t�t�l�k�p�t�t�t�t�t�t�t�t���'�@�M�X�Z�^�M�4�'������������� ^ s /  D ; @ A z + A M 6 G U G ) . + 4 , N D 3 G ) ^ \ = - > @ / L X W ^ P @ 1 O N H ? Q C [ [ N N )    �  �  �  �  b  �  �  w  �  x  c  f  w  w  9  M  {  9  �  %  �  �  �  �  �     i  �  B  �  �    �  ]    �  .  �  �  �  =  O  �  �  �  	  �  �  B  ~  U�C���<��<e`B;�`B<�9X<t�<��
<��
=m�h<�/=D��=e`B=�O�=�hs>	7L=0 �=aG�=D��>Kƨ>-V='�=D��=�+=��w=ix�=8Q�=T��=�v�=u=�
==�9X=aG�=]/=��=��P=�+=��=y�#=�j=u=��=�"�=��T=���=�-=���>��=�
==ȴ9>�-B*�B
LZB�BהB%�BtB��B	g�A��.B��B@�B��Br8B#SvB��B�\B4B [B"z�B_eBE�B|�BURB�tB}�B!�B��B��B�BL5B �B}B0��B��B��BU/B$��B��B��BRB?�B,Y`B�$B�HBk�Bx0B�:B�
Bf�BlB��B?�B
�B�YB�=B%��B<VB�JB	u?A�ReB�%BC�B�CB��B#?�B�EB0-B�B I�B"�&B@�B?�BCfB~jB�]B�wB"�B@�B @�B	B@�B�gB?�B0��B�aB�'B?�B$�wB�fB��BXbB8IB,@B�B}B�@B�B��BZB�lB��B��A'��A�`)A�k C�hr@�o�A�k�A'��An5'A�`�A<�{A .Ab>�A?Ip@�p�A���A�vaA߹�A� kAYHAAӰC��`AǨi@��A��A��%A�;2@{�@oA��A�v@���Aa�Al{<A|/�A>�CA���A=tAמnA�iyA��A��AC�KA�@A�`A��J@"�5@"�B ��A��@ʼ+A'��Aȅ�A�{�C�f�@��A��A(�Am'A��RA=��A �.Ab�AA6@�y<A��;A�A߄�A�{�AZ��A�C��A�[�@��A�mDA���A�hU@}�@{�+A���A�s@��A`��Ak�nA|(�A>��A�ehA<�$A�q�A���A��3A�vA �C��A��;A��A���@$*@�VB �A�}i@��L         %                     *   	      #   .   -   i            �   �            '         
   2      <   )                        %         *            
   K      
   A                  #                     %   +   /   )            3                           #      '   #                        )                        %                           #                     !                                                         #                        )                                 N�{�NPIN���N���NO�1OѪ�NZF�NM�{NU2�O@a�N9�AO��O�oRO��O���O�U9Nn�HO��O\�O�&O&!�Nr�N1P�O$sO�xFN�HN/a�N��AO��wOB�=O�,O�v�N��oN50NC�xN��VN�= N���N��P-�%N ��N_}6O)��N��RN���N�aTN��@O��hO��NP;iO��y  �  x    '  V     �  �  �  w  �  �  w  �  ?  �  �  �  �    �    h  _  W  �    �  S  �  y  �    1  ]  @    [  �  4  �    
�    �  [  �  
�    	  o�'t�<49X%   :�o;ě�;�`B<e`B<�o<�h<���<�1<�`B=D��=#�
=��<�=o=+=��=�1=\)=�P=#�
=8Q�='�=0 �=0 �=m�h=D��=�7L=D��=D��=P�`=ix�=]/=]/=aG�=aG�=e`B=ix�=�\)=�+=�O�=�\)=�hs=��P=ȴ9=�1=�E�=�Q����������������pt{�������vtpppppppp,,-/<HNUVUQH=<:/,,,,_[\ahnqz���zunka____#&08:610-#������% �������������������������[[`grtz}ytg[[[[[[[[["&/;C;8/""MHHJOS[chtw��{lh[OM346BOVOOB63333333333�������������nkpy��������������tn���������	

����������� ���������� )047751)�0,05BILHB50000000000kihlz������������zmk���������� ������������)5JMJC8)��������

��������������������������fhqt�����tohffffffff879:@GHUajnomdaUKH<8�������	���������������������������oirt������ztoooooooo���������������������������
""
����������������������������)BHOQQND6)�������
#)+*#
��)*+26COSSQOIC6,*))))XT[htvzth[XXXXXXXXXX�"�����


��������
#$$$$##
�����������������������
 ###
�����ee[it�����������{toe	
#)#
��������������������������
����-)*./<HFF></--------����������������������������������������')*6765)yyz���������������|y������

�����%)-41)�}z|�������������������Ľɽнܽݽ�ݽнĽ��������������������zÀÇÌÇÃ�z�n�m�i�n�o�z�z�z�z�z�z�z�z��������	���������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eͻx�������������x�l�d�l�u�x�x�x�x�x�x�x�x���������%�&�2�5����������žž�����߽��Ľнݽ��ݽнĽ����������������������y�y�������������y�q�n�y�y�y�y�y�y�y�y�y�T�a�b�j�m�q�s�m�e�a�[�U�T�Q�T�T�T�T�T�T�4�A�M�Z�a�f�n�s�u�t�s�f�Z�M�A�(�$�(�/�4�ʼּټ�߼ּʼʼ��Ƽʼʼʼʼʼʼʼʼʼʿ�"�.�;�G�T�`�i�v�p�d�`�G�.����������A�M�Z�f�n�����������s�f�Z�D�3�,�-�;�A���������������������������}�z�r�r�}����
��#�+�&�"����
�������������������
��#�0�7�<�C�B�9�0�#��
�������������
čĚĦĳıĦĚčĊĊččččččččččŭŹ��������������������ŹŭŨŠřŖŞŭ�����	���&�&�!��	�������޾߾������)�/�3�2�.�&�������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��a�n�zÇÊÇÀ�z�n�a�Z�W�a�a�a�a�a�a�a�a�����ûŻûû����������������������������a�m�z�������������������z�m�k�d�a�_�_�a����������������������������������������ÓàìñùþùùìàÓÇ�ÁÇÈÓÓÓÓ�-�:�F�I�J�F�F�:�4�9�-�)�-�-�-�-�-�-�-�-���!�-�3�:�F�F�:�:�-�-�!��������	��"�/�=�E�J�I�@�;�/�"��	�����������	�����������������������������������������лܻ��������������ܻлû��������п"�;�T�\�`�u�x�l�`�G�:�.�"�������	�"�m�q�y�����������y�m�i�`�]�\�`�g�m�m�m�m�ѿݿ���ݿѿ̿ʿϿѿѿѿѿѿѿѿѿѿѾM�Z�f�f�g�f�^�Z�M�M�K�L�M�M�M�M�M�M�M�M���������	�	����������������������������(�4�A�M�Z�e�f�s�~�y�s�f�Z�M�A�A�4�(�&�(�)�6�B�G�H�B�A�6�*�)�'�(�)�)�)�)�)�)�)�)���&�#�!���������������
���������������������������s�O�D�@�G�Z�i����ĦĳĴĸĵĳĮĦěĢĦĦĦĦĦĦĦĦĦĦ���������������������y�v�y��������������E�E�E�E�E�E�E�E�E�E�E�E�E�EtEiEiElEuE{E��B�N�[�g�j�p�g�[�N�B�>�<�B�B�B�B�B�B�B�B��(�5�A�N�U�N�E�A�5�(����������������������������������������������������������������������������������������������ºɺ˺ɺǺĺº����������������������6�8�C�M�O�V�\�`�\�[�O�C�6�.�*�%�!�*�3�6�t�t�t�l�k�p�t�t�t�t�t�t�t�t���'�@�M�X�Z�^�M�4�'������������� ^ @    D ; @ A z & A M 2 ' * ' ) ( +   N B / 6 ) ^ \ 0 $ ? @ / L G W ^ P @ 1 O 2 H ? Q C [ P N N )    �  p  �  �  b  �  �  w  �  �  c  f  �  %  .  <  {    �  �  d  �  \  j  (     i  �  S  �  Y    �  ]  \  �  .  �  �  �  =  o  �  �  �  	  �  b  B  ~  U  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  L  �  �  F  Y  k  v  o  h  _  T  I  =  1  %      �  �  �  �  �  �  v  �  �  5  i  �  �  �  �  �  �  �  �  �  G  �  �  A  �  f  $  &  '  &  #      �  �  �  �  �  z  N    �  t    �  N  V  Q  K  F  :  -       	  �  �  �  �  �  �  �  �  �  �      	       �  �  �  �  �  �  �  �  �  w  L  &      )      �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �  �  �       �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  a  M  :  (                �  
  5  S  g  s  v  n  c  R  <    �  �  q  &  �  �  p  C  �  �  �  �  �  �  s  f  W  G  9  ,        �  �  �  �  �  �  �  }  r  i  ]  P  A  .    �  �  �  y  P  )  �  �  �  �  9  [  o  u  v  p  g  [  M  =  (    �  �  �  _    �  S  �    �      8  a  �  �  �  �  �  �  �  �  k  2  �  _  �  �  X  �  �  �    <  <  %  8  3  '        �  �  �  �  �    
�     U  v  �  �  �  �  �  �  �  M  
�  
t  	�  	(  /  �    �  �  �  �  �  y  _  E  )  	  �  �  �  y  A  
  �  �  �  �  �  �  �  �  �  �  |  ^  >    �  �  �  �  r  >  �  �  �  S   �  �  �  �  �  �  �  �  �  �  �  �  �  |  b  B    �  �  �  '  �  �  ?  �  =  �  �        �  �  6  �  �  �  
^  �  �   �  �  �  )  �  �  5  t  �  x  *  �  0  �  �  �  �  �  �  	�      {  w  s  n  i  e  `  ]  Y  ]  j  w  �  �  �  �  �  �  �  `  c  f  h  b  W  G  6  $    �  �  �  �  �  r  Q  0    �  T  ]  ^  W  I  6  !    �  �  �  l  7  �  �  Z  �  |  �  ?  ;  I  V  P  >  %    �  �    K    �  �  m  
  �  �  	    �  �  �  a  >    �  �  �  H    �  �  �  �  �  r  X  9  9      	      �  �  �  �  �  �  �  �  �  �  �  �  v  f  V  �  �  d  H  ,    �  �  �  �  l  @    �  �  �  R  !   �   �  <  D  I  L  O  Q  N  0    �  �  e  "  �  }    �    -  -  �  �  �  �  �  �  z  [  2  �  �  �  ^  T  I  3    �  �  �  �  Q  �  �  #  U  p  x  m  K    �  f  �  t  �    ?  m  �  �  �  �  �  �  Y  0  )  `  X  G  .  �  �  v  +  �  1  2  H            	    �  �  �  �  �  �  �  �  r  \  @       1  *  #          �  �  �  �  �  �  �  �  �  �  �  �  �  =  A  D  C  K  Z  T  H  ,    �  �    G    �  �  Y     �  @  4  %    �  �  �  �  �  m  1  �  �  w  1  �  �  :  �  m          �  �  �  �  j  H  #  �  �  �    L    �  �  o  [  P  E  0    �  �  �  �  \  +  �  �  S    �  Y  �  �  G  �  �  �  �  �  �  |  q  g  ]  M  8  "    �  �  �  P     �  4     	  �  �  �  �  |  i  W  >    �  �  �  o  8  �  �  �  �  �  �  �  �  �  �  �  �  s  d  T  E  ,  �  �  �  r  D    2  �  �  �  �  �  �       �  �  �  }  ;  �  �    z  �  %  
�  
�  
�  
�  
{  
S  
!  	�  	�  	P  �  |    �  
  �  �  H  p  �      �  �  �  �  �  �  b  F  "  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  h  Z  L  =  /      �  �         [  U  J  9  (      �  �  �  �  �  ]    �  �  M    �  e  �  �  �  �  �  �  �  �  l  S  ;  #    �  �  �  �  s  A    	�  
  
\  
�  
�  
�  
�  
  
`  
6  
   	�  	w  	  ~  �  �  �  Z  ;      �  �  �  u  J    �  �  �  S    �  a    �  ;    d  	  �  �  �  �  a  >    �  �  �  �  ]  8      &  ?    �  o  `  P  6    
�  
�  
u  
*  	�  	g  �  v  �  N  �  �  �  �  �