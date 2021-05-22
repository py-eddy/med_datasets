CDF       
      obs    9   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��S���      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       P��      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �}�   max       >	7L      �  t   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>���Q�   max       @F�=p��
     �   X   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��(�`    max       @v�\(�     �  )@   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @P�           t  2(   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ϋ        max       @�p�          �  2�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �ix�   max       >��\      �  3�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��@   max       B4\�      �  4d   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��\   max       B4Kw      �  5H   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >31|   max       C���      �  6,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =o
8   max       C���      �  7   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          K      �  8�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          =      �  9�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       P�J�      �  :�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�L�_��   max       ?䗍O�;e      �  ;�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �}�   max       >�-      �  <h   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��Q�   max       @F�=p��
     �  =L   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��(�`    max       @v�
=p��     �  F4   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @P�           t  O   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ϋ        max       @�           �  O�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         Fs   max         Fs      �  Pt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�kP��{�   max       ?�*�0�     �  QX                        �            9      7      	   )   _                  �   S   !         .   	         	      
                                 2   '      #         &   	      �      N��	NFt�OD^�N��N]DN9�O���P?��N��;ONHN��nP��O`��O�&�Nv��N�bO��JP{�N��O�K OkŨN��[N�=8P��WP��O��cO�>N"q.O�3O~�N�%NxG)N��cN�h�O/!�O
�O�0:N�n�O���N�)M���OA@`N0��O ��N˥MPL�O�aiO�P(hN��NE�QO�N��N>7
O��O�@N����}󶼣�
�T���49X�#�
�t��D��:�o;D��;�o;�o;��
;�`B;�`B;�`B<o<t�<t�<#�
<D��<T��<e`B<e`B<e`B<�t�<�9X<���<�/<�`B<�`B<�h<�=C�=t�=�P=�P=�w='�='�=49X=8Q�=@�=D��=D��=T��=T��=Y�=aG�=e`B=ix�=��=�hs=�hs=��w=�1>$�>	7L��������������������nqtz�����tnnnnnnnnnn����������������������������������������]^annz{����zna]]]]]]����� ������������������
#5:80)#
���&%5BN[t��������tgF-&����������������������	"/586/)$"	���)6=A>6,)��/Ha��������a3#
�geot�������������tg������������������������������������������������������������#-5BN[bitxztg[N>5,)#661<Banvz����unUG=6#)+6BOOOLIDB86))####��������������������.*%$03<IU`bdda]UI<0.��).4)'������`^^aefnz}�����~zuna`O[�����������hB����5BX]TVbp[)��#/<HPUWWRH<#
oooqt�������������to������������������������������������������������

������PZ[^ht����uth^[PPPPW_ammz}�}zmaWWWWWWWW0+,6@BOYWROBB6000000FNV[gt������tg[SNFF")/;HT`ada`TH>8/"���������

���UI<1#����
#/<DHU��������������������)16BGIE>=6)
#/0330.(#
�������������������������������������)+,)@=AHMO[hjorrmh[WOIB@yuvyz������������zyy���������	�������������������������������!(�������)5AEC5&���3/5BNQTQNBBB85333333;BINV[]gb[NB;;;;;;;;�����!%&	����?BBO[bc][ODB????????����������������������������

���������������	����������������������������ܻ�������������ܻлһܻܻܻܻܻܻܻܻ������ûλû�������������������������������$�*�-�2� ��������������������������������ùϹҹܹݹܹϹϹϹù��������������������� � � �������m�z�~�������z�m�d�l�m�m�m�m�m�m�m�m�m�m�_�x�����������������x�l�O�F�D�D�F�K�S�_�)�6�N�c�q�s�d�6�)����������������)��������������������¿²°²¹¿���������a�c�m�r�y�z�����z�m�a�T�H�;�7�4�<�H�T�aÇÓÜØØÓÇ�z�o�z�|�ÇÇÇÇÇÇÇÇ��������"�%�	�����������������\�Y�g���T�a�z���������������z�v�m�a�T�M�H�H�M�Tìù����������������ùÕÓÇÁÀÅÐÖì�����������������ݾھ۾�������#�/�4�7�4�0�/�#� ����#�#�#�#�#�#�#�#����������������������������������������FF$F=FCFJFYF[FVFOF1FE�E�E�E�E�E�E�E�F�ʼԼּ��׼ּּʼ��������������ʼʼʼʾ�(�4�A�M�T�X�_�\�Z�M�A�4�(��	�����f�r�������������������������q�d�_�c�f�M�Z�_�f�g�h�i�f�Z�N�M�A�K�I�M�M�M�M�M�M�������������������������������������Ѽ'�@�Z�_�g������Y�4��������������'Ƨ������B�I�B�9�����ƧƁ�h�S�%�*�hƁƧ�������Ѿ޾־Ⱦ�������s�l�h�g�c�f�s�������������������������������t�w�����������	��"�$�"���	����������������������ìù���������	��
�������������ùìèì��&�(�4�7�4�A�D�H�A�4�(���������@�L�R�Y�^�b�Y�V�L�G�@�3�3�/�3�7�@�@�@�@ŭŹź��������ŹŭŦţŨŭŭŭŭŭŭŭŭ�������������������z�x�y�z�������������������������������ݽݽ׽ֽٽݽ����#�0�5�0�-�-�*�#���
�����������
�� �#���(�+�5�B�N�Q�O�N�F�B�;�5�)������m�T�L�G�=�@�G�T�j�m�y�������������y�p�mFJFVFaFcFeFcF`FVFJF=F6F1F/F1F=FEFJFJFJFJ�e�~�����������Ǻͺ��������~�k�^�e�e�Z�e�M�Z�f�h�n�m�s�t�s�f�b�Z�M�F�A�;�A�A�M�M������������������������������������������������.�:�>�D�C�:�-�!��������'�3�8�;�>�3�'�#��$�'�'�'�'�'�'�'�'�'�'�-�:�F�S�_�a�h�c�_�S�F�:�-�'�!�!��!�)�-�����������������������������������������	�"�/�;�T�m�{�����z�m�a�T�;��������	�����Ľνݽٽǽ������������y�r�p�{�������!�:�F�S�T�S�M�F�:�-�!��	������ �!����(�?�H�R�R�I�A�5�(�����ݿʿ̿ؿ���y�����������y�m�g�`�`�`�m�p�y�y�y�y�y�y�U�V�a�i�n�o�n�a�U�O�N�R�U�U�U�U�U�U�U�U���
��0�Q�`�a�U�I�<�0�#�
��������ĵ����ݿ�������ݿѿϿʿѿӿݿݿݿݿݿݿݿ��H�T�]�a�b�a�[�T�H�E�D�G�H�H�H�H�H�H�H�HD�D�D�D�D�D�D�D�D�D�D�D�D�D�D~DzD|D�D�D�EiEuE�E�E�E�E�E�E�E�E�E�E�ExEuEqEjEiEeEi���ʼּ׼�����ּʼļ��������������� ; R B � U 8 4 / l : [ a G R W f F . /  * d 5 g S   < S @ @ = Q S a k G > 1 N O p = : " W X : ` " A p m  V + E ?    �  e  �  �  �  E  1  t  �  �  �    �  �  �  �  Z  N  �  
  �      �  $  �  a  C  7  g  �  �  �  I  �  :  )  �  <    G  �  Y  Y    �    q  �  �  �  s  �  [  v  X  �ixջ�`B%   �D�����
���
<D��>+<�t�<��<�C�=y�#<�9X=}�<#�
<�C�=L��=��<�`B='�=\)<�9X<���>��=���=]/=L��<�h=��=t�=L��=+=,1='�=@�='�=�%=q��=}�=P�`=L��=y�#=Y�=�O�=u=��`=�j=�O�=�^5=y�#=�\)=�/=��
=�E�>��\>$�/>(��B�B�kB�CBɫB��B�B$�B	^�BK�A��@B8=B~lB
��B!�B4\�B��BnHBD�B�kB )�B&�9B�0B�zB��B�zBS�B
̂B!�B��B�B�A�Z�B=�B	R�A�mYB;BɹBPbBL�B$��B?yB*�B�PB"=B 6�BClB+�B	VB}�BɔBEzB�B�IB�mB�B�)B�BB�;B��B�?BA+B0B?7B$��B	B?qA��\B��B�B
G�B!��B4KwB�fB@�B?�B�B @�B&@BB�.B�B�B��B�dB
ӱB!��B��BB6�A��B[�B	qA��B<�B�[B?�B?ZB$�B@�B*��B7B<�B ASB9�B+D5B��B��B��B��B�B��BFsB@yB�LB?�@��H@�;<A���>31|A���A�2$@��A֧�A��RA�AAɗ#A���A��ǍdAU�A���A��C���@��A8��@��VA>32A��n@�7=B��AI��A��A\cA�L�A6��?�A��A�-uA-�,A��A��Ajo#C���@A>�AJ@d��?��6@��5A��DA��A �@tsxA���Al�|A��A�JA}zA�.�C��C���A 
�@�eD@��UA�J	=o
8A��XA�Ŗ@��~Aք�A�bA�k:AɂA��A�g�A�~�AU�bA�A�~qC���@�YA9Z@�	oA?�A��0@��B�<AJ	LA��NA\��A�oPA6�?ș�A��oA�t
A-'�A��AA���Ai�C���@5�A=�gAI:@l �?�9Y@�
0A��A��@A"��@l�uA�kBAl�A�A�ȉA|��A�~C���C�A ��            	            �            9      8      
   *   _            	      �   S   !         .   	         	                                    	   3   (      $         &   
      �                               -            K      !            %                  ?   K   #                                                            /   #      )         '                                                   9                  #                     =   !                                                            /   !      )         '               N��	NFt�OD^�N��N]DN9�Oy��O���N��;N���N��nPm��O`��O:EHNv��NF�4O' �O�ŸN��O�OP�,N��[N�=8O�P�J�O�\�O�>N"q.O/��O~�N�%NxG)N��cN�h�O[O
�Oy�N^��Op��N�T|M���OA@`N0��O ��N˥MPL�O�D�O�P(hN��NE�QO�N��N>7
OJO�@N���  D  �  �    [  �    �  �  �  �    �  �  �  A  �  $  >  _  �  �  �  
�  R  k  �  �  �  �  �  B  L    �  �  �  �  �  k  c  �  j  X  �  �  �  f  �  P      �    #  	�  u�}󶼣�
�T���49X�#�
�t��o=e`B;D��<u;�o<�C�;�`B<ě�;�`B<#�
<�1<���<#�
<�9X<u<e`B<e`B=�"�=��<���<���<�/='�<�`B<�h<�=C�=t�=��=�P='�=49X=,1=8Q�=8Q�=@�=D��=D��=T��=T��=m�h=aG�=e`B=ix�=��=��P=�hs=��w>�->$�>	7L��������������������nqtz�����tnnnnnnnnnn����������������������������������������]^annz{����zna]]]]]]����� �����������������
#1930&#
����BADKN[gt������tg[PGB��������������������	"..'" 	)6=A>6,)�/Han�����zaU4#geot�������������tg������������������������������������������������������������45ABN[bimphg[NEB<654;;>Janw|����naUK@=;#)+6BOOOLIDB86))####��������������������+'&036<IU\bb_[UIG<0+��).4)'������`^^aefnz}�����~zuna`eaghpt�����������the���)5JOHGOUSB)��	#/<HNSVUNH<#
	oooqt�������������to������������������������������������������������

������PZ[^ht����uth^[PPPPW_ammz}�}zmaWWWWWWWW0+,6@BOYWROBB6000000FNV[gt������tg[SNFF%/;HRT^acaa_TH?;91/%���������

������
#/<CHLMH/#
���������������������	)36BFHD=<86)
#-,&#
�������������������������������������)+,)@=AHMO[hjorrmh[WOIB@yuvyz������������zyy���������	��������������������������������!(�������)5AEC5&���3/5BNQTQNBBB85333333;BINV[]gb[NB;;;;;;;;�����#$
����?BBO[bc][ODB????????���������������������������

�����������������	����������������������������ܻ�������������ܻлһܻܻܻܻܻܻܻܻ������ûλû�������������������������������$�*�-�2� ��������������������������������ùϹҹܹݹܹϹϹϹù��������������������� � � �������m�z�~�������z�m�d�l�m�m�m�m�m�m�m�m�m�m�_�x���������������x�_�R�F�E�E�H�N�S�[�_�)�6�B�O�S�Z�X�P�6�)�������������)��������������������¿²°²¹¿���������T�a�g�m�p�s�m�a�T�J�H�A�H�I�T�T�T�T�T�TÇÓÜØØÓÇ�z�o�z�|�ÇÇÇÇÇÇÇÇ���������
���	�������������������x�o���T�a�z���������������z�v�m�a�T�M�H�H�M�Tù������������ùìàÓÍÇÇÎÓàäóù�����������������ݾھ۾�������#�/�2�6�3�/�$�#�#����#�#�#�#�#�#�#�#����������������������������������������FF$F=FJFUFVFJF1F$FE�E�E�E�E�E�E�E�FF�ʼԼּ��׼ּּʼ��������������ʼʼʼʾ(�4�A�A�M�M�U�Q�M�A�4�(������!�(�(�r�������������������������s�g�f�a�f�r�M�Z�_�f�g�h�i�f�Z�N�M�A�K�I�M�M�M�M�M�M�������������������������������������Ѽ'�4�@�A�M�R�Y�Z�Y�Y�M�@�4�3�'�&�"�"�&�'Ƴ������&�5�/�������ƚ�u�h�S�C�MƁƧƳ���������ξپҾž�������s�o�k�k�s���������������������������������t�w�����������	��"�$�"���	���������������������������������������������������úôùþ�ž�&�(�4�7�4�A�D�H�A�4�(���������@�L�R�Y�^�b�Y�V�L�G�@�3�3�/�3�7�@�@�@�@ŭŹź��������ŹŭŦţŨŭŭŭŭŭŭŭŭ�������������������z�x�y�z�������������������������������ݽݽ׽ֽٽݽ����#�)�+�(�#�"���
��������������
���#���(�+�5�B�N�Q�O�N�F�B�;�5�)������m�y�������������y�n�m�`�U�H�?�B�G�T�X�mFJFVF\FcFcFcF\FVFJF@F=F5F=FHFJFJFJFJFJFJ�~�������������úʺ��������~�r�m�_�f�f�~�M�Z�f�f�l�k�f�Z�M�H�A�=�A�D�M�M�M�M�M�M������������������������������������������������.�:�>�D�C�:�-�!��������'�3�8�;�>�3�'�#��$�'�'�'�'�'�'�'�'�'�'�-�:�F�S�_�a�h�c�_�S�F�:�-�'�!�!��!�)�-�����������������������������������������	�"�/�;�T�m�{�����z�m�a�T�;��������	�������ȽֽԽ̽������������z�u�s��������!�:�F�S�T�S�M�F�:�-�!��	������ �!����(�?�H�R�R�I�A�5�(�����ݿʿ̿ؿ���y�����������y�m�g�`�`�`�m�p�y�y�y�y�y�y�U�V�a�i�n�o�n�a�U�O�N�R�U�U�U�U�U�U�U�U���
��0�N�]�^�U�I�<�0�#�
��������Ľ����ݿ�������ݿѿϿʿѿӿݿݿݿݿݿݿݿ��H�T�]�a�b�a�[�T�H�E�D�G�H�H�H�H�H�H�H�HD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�EiEuE�E�E�E�E�E�E�E�E�E�E�ExEuEqEjEiEeEi���ʼּ׼�����ּʼļ��������������� ; R B � U 8 ) * l 3 [ T G J W i : 1 /  - d 5 : T  < S 5 @ = Q S a T G @ ) R . p = : " W X : ` " A p l  V  E ?    �  e  �  �  �  E  �  t  �  �  �    �  �  �  �  |  �  �  =  �      4  �  �  a  C  t  g  �  �  �  I  U  :    c     �  G  �  Y  Y    �  �  q  �  �  �  H  �  [  C  X    Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  Fs  D  >  7  1  *  !        �  �  �  �  �  �  �  �  �  �  v  �  �  �  �  �  �  �  !  #    �  �  �  �  �  ]  4  �  �  �  �  z  r  k  [  J  7  %      �  �  �  �  �  �  u  E  	  �    �  �  �  �  �  �  �    *  4  *  !        �  �  �  �  [  Q  F  <  1  %              �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �            �  �  �  �  �  �  �  �  �  �  �  s  P    �  
H  $  �  y  �  d  �  �  �  �  �  e  �  [  �  
�  	�  2  y  t  �  �  �  �  h  I      �  �  �  �  �  d  %  �  �  t  `  �  	  >  k  �  �  �  �  �  �  �  �  �  W    �  A  �  k  �  y  �  �  �  �  �  �  �  �  �  �  �  s  D    �  �  y  H    �  �    A  f  ~  x  h  Q  &  �  �  
      D  (  �  }  �  �  �  �  }  r  g  ]  ^  ]  Z  T  J  =  .    
    �  �  �  �  �    Z  �  �  �  �  �  m  C    �  �  Y  �  o  �  "    �  �  �  �  �  ~  v  m  e  ]  U  N  F  >  6  -  %          -  5  =  @  A  >  9  5  0  0  5  1  "    �  �  �  �  U  )  �    C  c  x  �  �  p  R  -    �  �  *  �  G  �  6  W  {  �  �  #      �  �  �  j  %  
�  
'  	z  �  �  $  S  �  �  �  >  <  :  6  4  1  ,  )  $        �  �  �  �  X    �  �  �  �    8  N  Z  _  ^  X  L  ;  &    �  �  Y  �  �  (  �  �  �  �  �  �  �  �  �  �  \  '  �  �  r  F  1    �  �  =  �  �  �  �  �  �  �  �  �  �  s  c  S  C  3  !    �  �  �  �  �  �  �  m  V  O  J  <  )    �  �  �  �  �  y  [  W  \  G  �  :  m  �  	  	  	�  	�  
H  
^  
j  
x  
�  
l  
  	v  �    �  s  �    @  O  Q  D  $  �  �  L  �  �  }  �  R  �  ,  e  �  O  _  j  b  T  ?  %    �  �  ~  L    �  �  H  �  �  `  .  �  �  �  �  �  e  F  "  �  �  �  b  +  �  �  �  Y  H  @  �  �  �  �  �  �  �  �  }  w  q  j  b  [  S  L  D  =  5  .  &  `  �  �  �  �  �  �  �  �  �  L  �  �  8  �  Z  �  �  ,  �  �  �  �  �  �  x  o  e  \  R  G  :  -  !      �  �  �  �  �  �  �  �  �  j  P  *  �  �  �  i  3  �  �  �  �  T    �  B  8  -  #        �  �  �  �  �  �  �  �  �  �  r  `  N  L  J  I  B  9  -      �  �  �  �  �  |  h  =    �  �  j    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  �  �  �  }  o  _  K  7  $    �  �  �  �  _  5    �  �  �  �  �  �  {  o  c  V  J  >  2  &          "  5  I  \  �  �  �  v  Z  7    �  �  w  7  �  �  �  ^    �    �   �  �  �  �  �  �  �  �  �  f  8    �  �  T    �  �  E  �  �  �  �  �  �  f  I  .  E  :    �  �  �  W    �  f  )  :  �  ^  b  g  g  \  Q  E  8  +      �  �  �  �  �  _  <     �  c  O  <  )    �  �  �  �  �  �  n  V  <  "     �   �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  n  F    �  �  V   �   �  j  j  k  l  n  w    �  �  �    K  �      	  �  �  	+  	�  X  T  L  @  1      �  �  �  v  :  �  �  t  :    �  �  �  �  z  k  _  T  H  6  %  
  �  �  �  �  X  +  �  �  �  e  1  �  �  �  �  [  %  �  �  o  +  �  �  6  8  �  �  P  �  )  
  �  �  �  �  �  s  [  5    �  u    �  [  �     �  �  �  �  f  V  A  (    �  �  �  �  �  f  E  !      �  �  g  (  �  �  x  J  $    �  �  �  k  =    �  �  o  I  $  �  �    �  P  D  7  +        �  �  �  �  �  �  �  �  �  s  c  T  E          
  �  �  �  �  �  �  �  �  x  d  P  5    �  �        �  �  �  �  �  M    �  b    �  (  �  Q  �  A  \  �  �  p  ]  J  6  #    �  �  �  �  �  _  @       �  �  �      �  �  �  �  d  4    	  ,  9  :  :  :  9  9  7  6  4  <  �  K  �  M  �  �         �  �  �    �  O  w  ~  N  �  	�  	�  	m  	7  	  �  �  Q    �  �  H  �  �  .  �  1  �  i    u  l  c  X  M  8    �  �  n  '  �  [  �  i  �  ^  �  �  �