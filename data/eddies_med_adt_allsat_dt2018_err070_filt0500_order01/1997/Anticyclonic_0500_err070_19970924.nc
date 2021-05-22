CDF       
      obs    ;   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�G�z�H      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�G1   max       P���      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       >	7L      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��
=p�   max       @F�(�\     	8   p   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��
=p�    max       @vx          	8  )�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0�        max       @Q            x  2�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�e        max       @�O�          �  3X   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��t�   max       >5?}      �  4D   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�   max       B4�      �  50   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��I   max       B4?�      �  6   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?�vQ   max       C���      �  7   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?�Ԍ   max       C���      �  7�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  8�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          Q      �  9�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          G      �  :�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�G1   max       P���      �  ;�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�t�j~��   max       ?�}�H˓      �  <�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       >	7L      �  =|   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��G�{   max       @F�(�\     	8  >h   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�         max       @vw��Q�     	8  G�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @Q            x  P�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�         max       @�H`          �  QP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D   max         D      �  R<   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�4�J�   max       ?�|�Q�     p  S(                     K      ^      )            '   e   a         =   	                  8   �            
                         O         
      ^                  q                        #NLO3�'NXG�N�N��2OG�P��O���P���N�D}Oy8�N��N`��N�dO�

P@QvP���N�EzN'nP��N�
'N�N��.Ny�N�QO ��O�eP���Nw��N��2N0k�N��jO;+�P�O��'O�L+O��Nh#�Or?#O��O�(�N�0MN�Z�M�PD�SNL��O�MN}��O�N7EO���N[h�OşO��xM�G1NC?�N��5NB��O�Ӽ��ͼ�1�o�ě���o��o;o;D��;��
;��
;ě�<o<o<o<#�
<T��<u<u<�C�<�C�<�t�<�j<ě�<ě�<�`B<�h<�<��=��=�w=�w=�w=49X=49X=49X=49X=8Q�=8Q�=D��=L��=L��=]/=ix�=q��=u=u=u=y�#=}�=�%=�o=�7L=�O�=�O�=��T=�^5=��`>	7L>	7L�������������"#$09<IMOOJB<80#"eehqt����theeeeeeeeddgt����������ytngdd($%)5=BDNY[\[QNB5)((QZ[gt����������tmg\Q����5BQPB=;5#�������	"/;<>;;7/"	���+-H\as���
����|a<+zz��������������zzzz#/HUXY\^TH</#������������������ 

��������������������#%/<HUhlljaUH<#BL[k�����������zpQHB�����5BTRK54)�����&*/6:<<6-*��������������������������
!$+33-$
���]XUTUafmqstqma]]]]]] ##����������������������������������������///0<AB?<0//////////������
!"" 
����)'-.06BOTW_ba^WOB6.)p�����;4�����}np401;HJOOHHG;44444444������������	)*)&)6BORZ[POB62)_VW^aimz��������zma_�����/296:)��������)--' ���TQSV[ht���������th[T����������������������������������������!#0<IUZ`\UG<0#�������

������������()������2-/06BBOW[]`[XOB>622 !}�����������}}}}}}}}BNgt�����t[N)����������������������������������������srtx���������tssssss����
!#%()%#"	�9<BHUW\UH?<<99999999��������
$&& 
�����������������������������������������������)5:75)�������������������������()6BOPOLCB@6.)((((((		
!#&'''#
				�����������������������������������������F�S�Y�_�d�_�U�S�S�F�D�B�F�F�F�F�F�F�F�F�l���������������������������x�n�l�a�e�l�L�Y�\�e�n�m�e�Y�N�L�E�H�L�L�L�L�L�L�L�L���ʾ;Ҿоʾž��������������������������.�;�G�N�R�T�U�T�O�G�?�;�:�.�,�(�*�.�.�.������������������������������������������<�I�U�r�~�v�k�U�<�������������������T�a�m�q�s�v�u�o�m�a�H�;�/�!��"�.�;�H�T�N�Z�s�����������������s�f�Z�N�(�?�>�N������� ����ݿٿڿݿ������������ �"�%�$���������������������B�O�O�Y�O�B�6�4�6�:�B�B�B�B�B�B�B�B�B�B���������������������������������������ѻl�l�x�����������������x�l�l�_�[�_�l�l�l�m���������������m�a�T�H�3�"�!�/�Q�T�_�māĚĳĿ����Ŀĳčā�t�k�`�T�I�L�[�t�ā������!�$�4�D�<���Ƨ�*���������OƎ�̿m�n�y���������������y�m�d�`�`�\�`�k�m�mE�FFFFFFE�E�E�E�E�E�E�E�E�E�E�E�E��(�4�A�M�Z�|��������s�Z�A�(��������(ƧƳ����������������ƳƧƝƠƧƧƧƧƧƧ�	���	�����������	�	�	�	�	�	�	�	�(�4�A�I�M�T�T�M�A�4�+�(�����(�(�(�(�׾���� ���������ݾ׾Ծ׾׾׾׾׾׼r�����������r�n�n�r�r�r�r�r�r�r�r�r�r�����������������������������������Ҽr�����ʼּ��ݼּ������r�f�Y�N�N�Y�f�r��
�����º��������u�o�w���������ߺ���#�0�3�3�0�#��
�
�
������������������ĿʿͿĿ������������������������ݿ��������������ݿֿݿݿݿݿݿݿݿ����
�������
����������������������ŔŠŭŹż������������ŹŭŠŔœŐőœŔ�������	�/�R�R�;�6�	���������������������z�����������������������z�n�a�X�\�Y�m�z�M�Z�f�s�����������������s�f�Z�E�>�C�M�ݿ������%�+�+�$����ͿĿ��Ŀʿѿݿ����������������������������������������������#�!����������ټѼҼּ��DoD{D�D�D�D�D�D�D�D�D�D{DqDVDIDADIDVDbDo�y���������������y�q�l�`�Z�X�S�Q�V�`�l�y�ѿݿ�������������ݿѿɿĿ����Ŀ̿ѿ��g�t�z�t�g�[�U�S�[�`�g�g�g�g����������������������������������B�[�h�uĖĘċ�z�[�O�B�6�+�*�/�+�����������������������������������������������������"�$�+�$��	�����������������˻!�-�5�:�>�:�6�-�!������!�!�!�!�!�!�`�m�y�����������y�u�m�`�G�;�7�;�F�G�T�`�g�i�g�e�\�[�N�F�F�N�[�[�g�g�g�g�g�g�g�gEuE�E�E�E�E�E�E�E�E�E�E�E�E�EuElEhEnEsEuìù����������������������ùìèìììì�H�U�a�n�zÂÇÓ×ÓÈ�z�a�U�H�?�9�<�F�H�#�/�6�3�*�#���������������������
��#������
���
���������������������������������������ܻܻܻ���������������������������������ĿĳħĬĳĿ���������̼@�F�J�@�?�4�'�!�'�(�4�9�@�@�@�@�@�@�@�@�� ���������������������� A 3 C H C = ) $ T V * K C j @ . r 0 J 2 U j C 8 I ! X O 2 : X I , E 0 U , 8 : x e 5 L � J 7 \ 2 X b * | Z E q V [ U X    <  �  w  "    �  /  4  	   	  �  =  m  �  W  [  
  �  >  �  �  N  �  �  -  [    �  y  �  k  8  �  �  G  f  l  u  �  �  ~    �  |  �  \  x  �  }  u  {  �  �  �  S  �  �  n  }��t��o;�o%   ;D��<�t�=��-=+=���<�1=@�<T��<D��<�o=D��=�l�=�`B<��
<�9X=��w<���<���=�P<�h=o=]/=� �>)��=49X=<j=,1=D��=�C�=��P=���=��=�o=D��=�C�>�=�7L=�O�=�+=}�>�-=�7L=�1=��=��
=�O�>5?}=��P=�{=Ƨ�=�1=��=��m>�+>,1B�B%�(BDHB
9�B��B	��B��A�BQ�B�B�.B��B��B^|B �B �HBMEB/��Bw�B$�A��EB��BC�B4�B&�B��BPB �A��B'�B�B��A�L�B��By�Bg�B��Be�B&�B6B-��Bb�B�nB)x�B��B�KBr!B�KB~B>�B�:B~8B�vB�*BC�B��BrEBL&B��B��B%��BAB	�WBP�B	��B��A��IB��B@�B�wB�B�B@�BB�B �]BA�B/�OBA	B$C�A�|�B�yB;�B4?�B&<�B\�B@kB�(A���B%�BŇB�A���B��B�jB<?B=�BsCB%HB@VB-8�BM�B�nB)�DB�B�<B�^B�`B��BR(B�B@B�XB��B~�B=AB=aBE�B��@���@��C?�vQAN�Ac<�A���A�kA��A�T6A��oA�I�Aؽ�A�ӵ@�mOA��A� NBw]Al��C���A9��B�9AX��A9xAV]u@�u�A���@�H�@0�A�<Av8�AL�A�UA�{UA��BA���AC}�A��"Ar5A��C��%A^\A|��A�R@]\�AَUA�1�A��\@o��Aj��A�i�C�`A�ӠA���A�|�A�i�@�lA�uX@ξ@�#�@�@�
�?�ԌAOp6Ac1A���AꁲA��A��4A��AҏDA؀�A�\�@���A��A�w B�-AlQ�C���A:8�B�DAX/�A:{_AW�r@�XA���@�
@4+A�uAu8+AqA� A�(A�-CA�a�AA+(A��Aq15A �C��A��A{��A�\�@Z��Aڀ�A�i�A��@nCxAl��A��C�<A�1�AƒQA�/�A��1@��WA�{)@Ά�@���                     L      _      *         	   '   e   b         >   	                  9   �            
                         P         
      _                  r                        #                     5      O                  '   -   Q         -                     !   =                  +                                 1                           !                                    '      =                        G         !                        %                  !                                 '                           !               NLN�uN3�N�N��2OG�P��O8��P��IN�D}O�N��N`��Nrf�O�wiO�{�P���N�EzN��O�b�N���N�N��.Ny�N�QN���O$�>P�Nw��N��2N0k�N��jO;+�O͌�O��'O�L+O���Nh#�Or?#N�C�O�(�N�0MN�Z�M�P�NL��O�MN}��N��AN7EOJ�N[h�OşO��xM�G1NC?�N��5NB��O��  �  �  �  �  }  �    _    �    .  (  e    ]  o  �  �  �  7  �  T  |    �    
�  �  N  �    �  :  �  m  P  @  �    �  �  �  �  ?  h  �  �  �  �     �  }  �  �  �      	���ͼ�o��`B�ě���o��o<ě�<49X<�;��
<�C�<o<o<t�<���=<j<�`B<u<�t�=�P<���<�j<ě�<ě�<�`B=\)=P�`=���=��=�w=�w=�w=49X=L��=49X=49X=@�=8Q�=D��=��
=L��=]/=ix�=q��=��T=u=u=y�#=�7L=�%=ȴ9=�7L=�O�=�O�=��T=�^5=��`>	7L>	7L�������������" #%/0<AIIIIB<80#""gfhrt����}thggggggggddgt����������ytngdd($%)5=BDNY[\[QNB5)((QZ[gt����������tmg\Q�����)7=>85)����		"//7542/,"	^Z`n����
������zg^zz��������������zzzz!#/<HMSUUUJH</'#!!������������������ 

��������������������!%&+/<HUaggdaUH<*$%!eadmz������������zme���)5BKLG2+������&*/6:<<6-*�������������������������
#')*(% 
����`ZVVXaemprsoma`````` ##����������������������������������������///0<AB?<0//////////�����

�������0-.156=BIOSWXXWROB60��������������������401;HJOOHHG;44444444������������	)*)&)6BORZ[POB62)_VW^aimz��������zma_���#,/2..)���������)--' ���TQSV[ht���������th[T����������������������������������������!#0<IUZ`\UG<0#�������

	 ������������()������2-/06BBOW[]`[XOB>622 !}�����������}}}}}}}})BN[fo|~{t[B)����������������������������������������srtx���������tssssss���	

"#$##
���9<BHUW\UH?<<99999999�������

�����������������������������������������������)5:75)�������������������������()6BOPOLCB@6.)((((((		
!#&'''#
				�����������������������������������������F�S�Y�_�d�_�U�S�S�F�D�B�F�F�F�F�F�F�F�F�x���������������������������{�x�t�v�x�x�L�Y�Z�e�j�j�e�Y�P�L�F�K�L�L�L�L�L�L�L�L���ʾ;Ҿоʾž��������������������������.�;�G�N�R�T�U�T�O�G�?�;�:�.�,�(�*�.�.�.������������������������������������������#�<�I�Y�`�c�`�U�I�0�����������������;�H�T�a�j�m�m�o�n�m�a�T�H�;�3�/�)�,�/�;�g���������
�����������u�o�n�h�O�L�Z�g������� ����ݿٿڿݿ����������������������������������������B�O�O�Y�O�B�6�4�6�:�B�B�B�B�B�B�B�B�B�B���������������������������������������ѻ_�l�x�������������x�q�l�_�]�_�_�_�_�_�_�z���������������o�a�T�H�A�4�5�@�T�a�m�zāĚĦıĸĻĿľĵĦĚč�~�t�l�j�d�i�tā������/�1�&���Ƨ�C�����*�C�hƎ����m�n�y���������������y�m�d�`�`�\�`�k�m�mE�FFFFFFE�E�E�E�E�E�E�E�E�E�E�E�E��4�A�M�e�o�v�r�b�Z�M�A�4�(��������(�4ƧƳ����������������ƳƧƢƥƧƧƧƧƧƧ�	���	�����������	�	�	�	�	�	�	�	�(�4�A�I�M�T�T�M�A�4�+�(�����(�(�(�(�׾���� ���������ݾ׾Ծ׾׾׾׾׾׼r�����������r�n�n�r�r�r�r�r�r�r�r�r�r��������������������������������������߼r�������������������������r�f�a�\�e�r���ɺֺ����������ɺ�������������������#�0�3�3�0�#��
�
�
������������������ĿʿͿĿ������������������������ݿ��������������ݿֿݿݿݿݿݿݿݿ����
�������
����������������������ŔŠŭŹż������������ŹŭŠŔœŐőœŔ���	�"�;�A�;�-�"�	�����������������������z�����������������������z�n�a�X�\�Y�m�z�M�Z�f�s�����������������s�f�Z�E�>�C�M������"�)�(�"�������ݿпʿпݿ�������������������������������������������������#�!����������ټѼҼּ��D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DtD{DD�D��y���������������y�q�l�`�Z�X�S�Q�V�`�l�y�ѿݿ�������������ݿѿɿĿ����Ŀ̿ѿ��g�t�z�t�g�[�U�S�[�`�g�g�g�g���������������������������������6�O�[�h�xČĎċā�t�[�O�B�;�;�-���&�6��������������������������������������������������"�$�+�$��	�����������������˻!�-�5�:�>�:�6�-�!������!�!�!�!�!�!�m�y�{�������}�y�m�h�`�T�Q�P�T�Y�`�g�m�m�g�i�g�e�\�[�N�F�F�N�[�[�g�g�g�g�g�g�g�gE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�EwErEvE�E�ìù����������������������ùìèìììì�H�U�a�n�zÂÇÓ×ÓÈ�z�a�U�H�?�9�<�F�H�#�/�6�3�*�#���������������������
��#������
���
���������������������������������������ܻܻܻ���������������������������������ĿĳħĬĳĿ���������̼@�F�J�@�?�4�'�!�'�(�4�9�@�@�@�@�@�@�@�@�� ���������������������� A / F H C =  % K V ' K C f A % s 0 I ( S j C 8 I " B - 2 : X I , 8 0 U % 8 : T e 5 L � ? 7 \ 2 C b - | Z E q V [ U X    <  �  [  "    �  �  �  0  	  E  =  m  �  ~  �  �  �     o  �  N  �  �  -  �  m  P  y  �  k  8  �  �  G  f     u  �    ~    �  |  }  \  x  �  �  u  �  �  �  �  S  �  �  n  }  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  �  �  �  �    s  h  ^  S  G  ;  .         �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  p  [  H  4    �  �    �  �  �  �  �  �  �  �  �  �  �  s  A    �  �  Q    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  y  v  q  i  a  U  E  5  $      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  f  Q  ,  �  �  T    <  .  �  �  �  �          �  �  t  2    �  �  :  �  �  �  o  �    @  R  \  _  W  B  -      �  �  �  e  ,  �  v  �  {  d  �  �  �        �  �  �  F  �  �  (  �  $  x  �  o  �  �  �  �  y  [  8    �  �    I    �  �  �  R    �  W       m  �  �          �  �  �  v  ;  �  o  �  O  �  �  �  .      �  �  �  �  �  z  e  O  9  '      �  �  �  �  �  (      
     �  �  �  �  �  �  �  �  �  �  �  r  b  S  D  c  d  e  V  >  $    �  �  �  t  B    �  �  �  �  y  ^  B  �  �  �  	      
  �  �  �  �  �  [  )  �  �  t  �  �   �  	s  

  
t  
�    H  ]  S  @  &  
�  
�  
E  	�  	#  V  =  �  P  �  �  K  k  n  `  >      2  �  �  �  �  �  �  R  �  �  �   �  �  �  �  �  �  �  |  u  n  g  `  X  P  H  @  7  *       �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  i  S  9      �    Z  }  �  �  �  �  �  �  �  ~  H    �  F  �  *  \  j  T  /  2  6  5  2  -  %        �  �  �  �  �  q  K  $  �  �  �  �  �  �  �  �  �  |  p  d  X  K  ?  2  #       �   �   �  T  S  Q  M  @  1     	  �  �  �  �  s  E    �  w     �  x  |  |  |  |  |  |  |  |  x  l  `  T  G  :  ,       �   �   �              
                            �  �  �  �  �  �  �  �  �  �  �  ~  B  �  �  c    �  m    5  w  �  �  �  �  �        �  �  �  =  �  #  [  y  �  b  c  	!  	�  
  
i  
�  
�  
�  
�  
�  
�  
�  
L  	�  	R  v  R  �  ;  �  �  �  �  �  �  �  �  �  �  �  �  w  i  \  P  C  .  �  �  a  N  C  8  ,      �  �  �  �  �  �  d  F  -           2  �  �  �  �  �  �  �  �  z  o  c  X  M  F  H  I  K  M  N  P      �  �  �  �  �  �  �  �  �  y  b  K  3    �  �  �  �  �  �  �  �  �  �  z  P     �  �  �  ]  (  �  y  �  6  v   �    !  ,  7  9  2  +    �  �  t  �  �  �  n  -  �  E  �  {  �  �  �  �  �  �  �  i  S  ?  #  �  �  �  a    �  �  �    m  \  K  >  6  /  '      �  �  �  j    �  t  2  �  [  3  3  L  P  J  <  .       �  �  �  �  �  w  J    �  �  5    @  9  1  *  #           �  �  �  �  �  �  �  �  �  �  x  �  �  �  �  �  �  �  �  u  `  K  5    �  �  �  �  I  �  +  u  j  V  �  �  �        �  �  )  �  �  �  
�  	h  �  �  �  �  �  �  �  �  �  �  ^  8      �  �  �  ]  1  �  g  "   �  �  �  �  �  �  �  y  k  W  8    �  �  �  d  $  �  �  C  �  �  �  �  ~  n  Z  D  .    �  �  �  �  n  G    �  �  �  �  �  �  �  �  �  �  ~  o  \  J  7  %    �  �  �  �  �  �  u  
�      0  >  8  1    
�  
�  
e  
  	�  	#  �  �  �  �    �  h  ]  R  G  =  2  (          �  �  �  �  �  �  �  �  �  �  �  w  l  ^  H  /    �  �  �  �  S    �  �  �  W  �  	  �  �  �  �  �  �  �  �    }  {  x  v  s  p  m  j  g  d  a  V  u  ~  �  �  �  �  �  d  :    �  �  e  )  �  �  v  4  0  �  �  �  �  �  �  �  �  �  �  �  �  |  v  r  m  o  �  �  �  �  �  g  �  �  �  �  �  �  �  !  �    0  �  c  
�  �  =  �  �  �  �  �  �  �  r  ]  G  0      �  �  �  (  �  b  �  �  }  j  ]  O  A  1    �  �  �  |  C    �  n    �  F  �  k  �  �  �  �  e  5    �  �  f  2    �  �  �  K  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  `  L  6      �  �  �  y  Q  '  �  �  �  �  m  ]    �  �  Y    �  �  <  �  �    *  �  y    �  X  �  �      �  �  �  X  /  
  �  �  �  �  �  N    �  d      �  N  	�  	�  	�  	�  	n  	0  �  �  ]    �  �  9  �  \  �  �  �  1  s