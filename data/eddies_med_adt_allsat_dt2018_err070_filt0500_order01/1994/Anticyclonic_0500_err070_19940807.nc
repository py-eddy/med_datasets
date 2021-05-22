CDF       
      obs    7   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�1&�x�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M� �   max       PD�j      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �Y�   max       =��      �  d   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�\(�   max       @F�\(�     �   @   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��(�\    max       @vnfffff     �  (�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @R@           p  1p   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @Ϊ        max       @�          �  1�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���   max       >w��      �  2�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B6      �  3�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��|   max       B68�      �  4t   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�C�   max       C�%�      �  5P   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��   max       C�)      �  6,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          1      �  7�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          %      �  8�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M� �   max       O�X�      �  9�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�=�K]�   max       ?�����      �  :x   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �P�`   max       =���      �  ;T   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�Q��   max       @F�\(�     �  <0   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��\(�    max       @vl�����     �  D�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @R@           p  M`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @Ϊ        max       @�K�          �  M�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?�   max         ?�      �  N�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�:)�y��   max       ?�4�J�     0  O�            $   7   
   	                                 	            $   �         F   (      ,   )                     }            
      K            0   
   J   F      	         >N�	*N�0'N��+O���P P�N��N��NF�On�O�m:O���N(��O�9OA�ENØ�O�,�N��N�c�N���O�vN��NO]GPC�O��O��9PD�jP,G,O��
O��[O�M� �Nyw�Nh�rN-NN��5O
PO�(N��OR$ANf�N�^�O!�cP
;}O�s\Nc�[NS�O�2}N�X�O��@O�Q�N�ѢN��?N��DNc.�O�(h�Y���h��o�e`B���
��o��o%   %   ;o<o<t�<#�
<#�
<T��<T��<�o<�C�<�C�<���<�1<�j<�j<�/<�/=o=o=+=\)=\)=�P=��=#�
=,1=0 �=0 �=49X=<j=@�=L��=Y�=Y�=]/=aG�=y�#=�C�=�O�=�O�=�t�=���=��-=��T=�E�=�v�=��egjt�����tkgeeeeeeee��������������������������������������'0<@LNUaw}���wnbI<0'�������
! 
��������������������������gdddinu{}~}}{ngggggg��������������������lnv{����������{rnnll����)5EID@72)��������������������������������������IGKPW[h�������tlh[OI��������������������"/;HILHC;/*"()5=BNSXZNB5)������

��������846:9<?HUY\\URKHC<88<768<HOU[[VUH<<<<<<<����$#$!������)55?75)"kdacmw����������zymkIEGQXet����������g[I(),25BNNRVZ\[NB95,)(MLOS[_cgt������tg[RM����
%<L\aa^U<���GCDJ=Hanw��������zUGvu����������������~v����"&+*'����������'/.)�����		

								"##/5<G</#�����

������������
####"
�����������������������������������������������������

����������������������������
"'&""
������^]chot||ztth^^^^^^^^10245BENPSONB5111111eefhiot����������the������������������136423)
��������������������1-6BHOOOKEB@86111111;633/05;HTaiig]TMH;;!),6BGB@<665)#mnz��������������zm���
 (./-#
���������������������������fgkt���������|tgffff��������������������Zamz����zmeaZZZZZZZZ==?DO[hmtwvstrmh[OB=�n�v�z���z�n�a�]�Z�a�h�n�n�n�n�n�n�n�n�������������������	���������������������������������������������߻��������лܻ���ܻ������x�j�b�o�{�����4�A�E�i�t�v�k�\�E�4�������������4��*�6�C�D�O�T�O�G�C�6�*�"��������Ľнݽ���������ݽнĽ����ĽĽĽĽĽ����������������������������������������Ҽ������������������������������������������	��"�6�G�F�;�.���׾��������ʾ��������������������������������������������"�*�"�!���	��������	���"�"�"�"�"�"�A�M�Z�f�����������s�f�Z�H�;�4�-�(�2�A�ûƻ̻лۻػͻû������������������������a�m�t�x�z�p�m�a�Z�T�Q�S�T�W�a�a�a�a�a�a�hƁƎƧƩƥƚƗƉƁ�u�h�]�V�N�L�R�U�\�h�����������������������z�}������ÇÓàìù����������ùìâàÓËÇÅÇÇ�������������������������������������5�N�Z�`�c�Y�N�A�5�������������5�Z�f�i�k�m�f�f�Z�M�K�K�M�O�V�Z�Z�Z�Z�Z�Z������������������������������ƿ�����������6�B�O�W�^�^�Y�O�6���������������¦ª°¦¥�t�o�o�q�t�v���)�5�G�N�[�c�_�W�B�5�������������(�4�M�s����������f�Z�A���	��������������"�1�%������������������������(�4�A�E�M�W�R�M�?�4������������(�;�G�W�_�b�\�T�G�;�.��	����� �	��"�;�/�;�H�T�\�Y�S�H�/�����������"�&�,�/�׾�������������׾Ծ׾׾׾׾׾׾׾��������������������������#�/�4�9�/�&�#�������������m�n�o�m�i�`�U�T�G�G�G�M�T�]�`�j�m�m�m�m�����������������������������������������m�y���������������y�w�m�`�U�T�S�T�V�`�mD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DxDzD�D�D������ʾϾ׾ؾ׾ʾ������������������������������ú����������~�v�r�e�]�`�e�~�������!�-�:�@�F�K�F�:�-�!� ��!�!�!�!�!�!�!�!���$�0�9�3�0�)�$���
�������������������ĽɽʽĽ��������������z�y�����U�^�a�`�\�T�F�#��
�����������
��#�A�U��#�0�<�E�I�M�N�I�<�0�#�
����������
��y�����������y�l�i�l�p�m�y�y�y�y�y�y�y�y�����
����������غ��������������čĳĿ����������#��
������ĳėčāĀč�����������������������������~�u�~����������������������ܹϹù������������Ϲ߹�E�E�E�E�E�E�E�E�E�E�EuEqEtE�E�E�E�E�E�E�ÇÓÙàçêëàÓÇ�z�x�u�v�zÀÇÇÇÇ�������������������w�s�p�r�s�w����������Y�f�r���������������s�r�p�f�Y�W�M�Y�YÓÞÜÝÖÓÑÇÀÄÇÏÓÓÓÓÓÓÓÓ�����'�?�O�Y�M�@�4�'�������ܻܻ�� / K > E 2 ; S :  R 9 | ; 5 A ' ' \ , . `   4 9 + A Q ) o p W 4 � i L $ Q F J 3 5 R 8 S N s Y > G $ a c ` N    �  �  �    E  �  �  X     e  (  �  7  �  �  f  �  '  �  1  �  �  &  A  U  U    n  �  �  :  �  m  �    F  d  B  �  �  �  W  �    �  f  8  �  #  �      �  �  ]���ě��t�<�9X=L��;ě�<o;�`B<�o<�h<ě�<D��=+<���<ě�=t�<�`B<���=t�=t�<�/=m�h>w��='�=ix�=��`=�hs=0 �=���=���=,1=<j=D��=8Q�=@�=e`B>-V=D��=��=ix�=�%=��P>o=��-=�o=���==���>z�>t�=Ƨ�=�Q�=�
==ȴ9>'�B	�B\�B�B'�fB#cqB*�B([!B�B)E%B�MB�B{�B��B"��A���B�qB#�mB(�B*�B�B�;B 9B
OfB�GB	�:BzB�B�BƴB4lB�cB�]B"&B�QB=�By�B�)B6B$6B�fB�(B�~B±B+eB,�iB"�A���B��B1B�B!�:B
DB]yA���B;�B	��B��B�yB'�B#BqB0�B(?FB�B)>B?�B\�B�B�#B"ƹA��|B�B#�QB@�B?BfxB��B BnB
@'B�B	��B>�B� B��B�B?�B��B��B?�B9B=�BA�B^B68�B$�B��B�B@�B�#B0nB,o�B?�A�RQB@GBH�B��B!�B
K�BA�A�ZjBivAǁ*A��bA�[S@��A7�/B cA*uAІU@�AZyA���A�!�A@��@��4A���B��@�ޗA�E�A�5�A�B�A??�B Aՠ/A���A�f�A=|A�9�A7��A`�A��AU�A�� A�xAgUArĩAl9�C�ߡAO��@�u@v��B	n{A!rA��vA�loA0�@P�HA�v�@R�>�C�C�%�A�(�A��}@��uA�yV@úTA�~0A�|A�p�@���A8��A�~�A*�AЂ�@���A[�A��vA�˘AB�@���A��kBS�@���A�}�A�{fA�^�A>��B=�A՜jA���A�ܯA>�AA��BA8�Aa GA��AUqA���A��Ah�gAr��Am C���AP�W@e�@tHB	;%A"tPA�Y�AꈖAt@L�9A�`@�|>��C�)A�rA�~�@�@Aɷ�@ǈ�            %   8      
                                 
            $   �         G   (      -   *      	   	            ~                  K            1      J   F      
         >            )   '               %                              %         )         1   1   !   !   )                                       )            #      #                                                 !                              !                     !   !      %                                       !                                    N[��N�0'N��+O�^O�S�N��N��NF�N���O�ޓO���N(��O(�0O�N�'zO�,�N:�N�c�N@z`O�X�N~��O6�O��N��O9l�O��SO��O��
OE�O���M� �Nyw�NA�qN-NN��5N�L-N�_vN��OR$ANf�N�^�O!�cO��sOo'5Nc�[NS�O6P�N�X�O�l�Od��N�ѢN��?N��DNc.�O���  �  �  �      �  �  �  ~  F  �  N  �  L  �  C    �  �  S  u  �    �    `  u    �  �  �  E  E  �    i  9  
  \  @  �  ^  
�  �  �    	  J  )  �    G  �  �  ƽP�`��h��o�D��<D����o��o%   ;D��;�`B<o<t�<�C�<T��<e`B<T��<�t�<�C�<�9X<��
<�9X<��=���<�`B=t�=q��=8Q�=+=H�9=��=�P=��='�=,1=0 �=8Q�=�
==<j=@�=L��=Y�=Y�=�O�=e`B=y�#=�C�=� �=�O�=��T=�v�=��-=��T=�E�=�v�=�/fgmt�����tmgffffffff��������������������������������������F<=EIPUZbntwzztnbUOF�������

��������������������������gdddinu{}~}}{ngggggg��������������������vrx{{��������{vvvvvv�)5<CC?>:5)��������������������������������������NRWY[^ht�����tjhc[ON��������������������"/;GHJH@;//"()5=BNSXZNB5)����

�����������846:9<?HUY\\URKHC<8898;<HIUVUUH<99999999������#"# �����)25>5)kimpz�����������zsmk`\]afq�����������tg`.*-35BLNRUXXNB<5....XTU[agtx�������tgb[X#/>MNIGA</#
RMSXX_cz��������znaRvu����������������~v�����	! ��������#)..-)�����		

								"##/5<G</#�����

������������
####"
���������������������������������������������������

�������������������������������
"'&""
������^]chot||ztth^^^^^^^^10245BENPSONB5111111eefhiot����������the����������
��������	)25311)��������������������1-6BHOOOKEB@86111111989<CHTW^accaXTHC?;9!),6BGB@<665)#�}����������������������
!%'(%#
�����������������������fgkt���������|tgffff��������������������Zamz����zmeaZZZZZZZZ>>@EJO[hquuqrpjh[OB>�n�q�z�~�}�z�n�a�_�[�a�l�n�n�n�n�n�n�n�n�������������������	���������������������������������������������߻����������ûлܻ�ܻлû������x�m�t�x���4�A�M�U�^�`�[�L�5�(�����������(�4��*�6�C�D�O�T�O�G�C�6�*�"��������Ľнݽ���������ݽнĽ����ĽĽĽĽĽ����������������������������������������Ҽ����������������������������������������	��"�1�<�A�?�;�.�"��	���ľ��׾���	�����������������������������������������"�*�"�!���	��������	���"�"�"�"�"�"�M�Z�f�s�z���������f�Z�T�H�A�?�A�C�K�M�������ûлֻԻлʻû��������������������a�m�s�w�x�m�k�a�]�T�R�T�T�X�a�a�a�a�a�a�hƁƎƧƩƥƚƗƉƁ�u�h�]�V�N�L�R�U�\�h������������������{������������������ÇÓàìù����������ùìâàÓËÇÅÇÇ��������������������������������������(�5�N�X�_�a�W�N�A�5�����������Z�f�g�i�k�f�d�Z�M�M�R�X�Z�Z�Z�Z�Z�Z�Z�Z������������� ������������������������������)�6�B�I�K�F�B�6�)������������¦¨®¦£�t�p�r�t�x���)�7�B�K�R�N�F�B�5�)�������	��A�M�f�s�{������f�M�A�4�(�!��� �(�4�A�����������	�	��������������������������(�4�A�E�M�W�R�M�?�4������������(�.�;�G�T�U�W�T�O�G�8�.�"��	�����"�.�;�H�T�Z�W�T�O�H�;�/�"��������%�)�/�;�׾�������������׾Ծ׾׾׾׾׾׾׾��������������������������#�/�2�7�/�#�#�!������������m�n�o�m�i�`�U�T�G�G�G�M�T�]�`�j�m�m�m�m�������������������������������������������������������y�p�m�`�^�V�Y�`�m�y�z����D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D������ʾϾ׾ؾ׾ʾ������������������������������ú����������~�v�r�e�]�`�e�~�������!�-�:�@�F�K�F�:�-�!� ��!�!�!�!�!�!�!�!���$�0�9�3�0�)�$���
�������������������ĽɽʽĽ��������������z�y�����/�<�H�U�X�Z�Z�V�N�@�#��
������������/��#�0�<�D�I�L�M�H�<�0�#�
����������
��y�����������y�l�i�l�p�m�y�y�y�y�y�y�y�y�����
����������غ�����������������������
���
����������ĿĸķĿ�����غ����������������������������~�u�~�������Ϲܹ�������������ܹϹù�������������E�E�E�E�E�E�E�E�E�E�E�E�E�EExEzE�E�E�E�ÇÓÙàçêëàÓÇ�z�x�u�v�zÀÇÇÇÇ�������������������w�s�p�r�s�w����������Y�f�r���������������s�r�p�f�Y�W�M�Y�YÓÞÜÝÖÓÑÇÀÄÇÏÓÓÓÓÓÓÓÓ�����'�4�<�J�M�@�4�'�������߻޻�� , K > B / ; S : & J 9 | 3 0 C ' 0 \ . - c   . ) # - Q # c p W / � i F  Q F J 3 5 M 7 S N N Y * B $ a c ` I    p  �  �    [  �  �  X  �  �  (  �  q  B  �  f  V  '  U    �  H  =    �  o  �  n  �  ^  :  �  S  �    �  �  B  �  �  �  W    �  �  f  �  �  �  �      �  �    ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  �  �  �  �  �  �  �  �  �  m  ;    �  t  *  �  �  9  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  F   �   �  �  �  �  �  �  �  �  �  �  �  �  w  i  [  H  .    �  �  `  �  �  �  �  �  	        �  �  �  �  �  �  ;  �  y    �  :  �  �  �  �      
  �  �    -  �  i  �  �    l  �    �  �  �  �  |    �    z  p  e  Y  J  9  (      �  -  g  �  �  �  �  �  �  �  |  ^  >    �  �  �  �  s  R  1     �  �  �  �  �  �  �  �  �  �          
        !  +  4  \  n  w  |  |  w  q  h  ]  M  6    �  �  �  �  V  !  �  y    /  <  C  F  :  "    �  �  �  �  �  W  "  �  �  B  �    �  �  {  j  U  A  2  $      �  �  �  �  �  �  �  k  0  �  N  O  P  P  Q  R  R  P  L  H  D  @  <  7  0  *  #        �  �  �  �  �  �  �  �  �  �  �  �  p  @     �  i  0    �  '      4  F  :  ,    �  �  �  �  _  .  �  �  �  �  :  /  �  �  �  �  �  x  n  c  X  L  ?  1  #  �  �  �    T  '  �  C  ?  :  3  ,  !    �  �  �  �  �  s  R  (  �  �  �  �  g  �  �  �              �  �  �  �  u  T  3    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  Y  4    �  �  y  9  Z  v  �  �  �  �  �  �  t  a  K    �  �  =  �  �  Q  �  R  S  J  ?  8  9  ?  @  9  '    �  �  ~  E    �  x  -   �  o  q  s  u  q  l  g  _  W  N  ?  +    �  �  �  �  �  v  \  H  q  �  �  �  �  �  g  9    �  �  8  �  �  "  q  q  U  "  �  �    *  �  �  d  �  
    �  r  �  �  �  N  �  2  	�  �  �  �  �  �  �  �  �  �  �  q  Y  ;    �  �  �  N    �  >  �  �  �                �  �  �  �  @  �  �    �  F  �  d  �  �  
  ,  N  ^  _  P  =    �  �  W  �  1  )    �  �  �    3  R  h  s  r  `  G  E  S  g  Z  -  �  ~    [  C    
  	  
  	    �  �  �  �  �  �  �  �  y  ]  =    �  �  �  @  d  �  �  �  �  �  �  z  ]  ;    �  �  a  �  =     �  �  �  �  �  �  m  A  8  !  
  �  �  �  r  5  �  j  �    @  �  �  �  �  w  X  8    �  �  �  �  �  b  D  &     �   �   �  E  ;  1  (           �  �  �  �  �  �  f  L  2    �  �  A  C  E  D  B  @  ;  6  %    �  �  �  y  G    �  �  e  *  �  �  �  �  l  W  C  0      �  �  �  �  �  �  �  �  �  t      	  �  �  �  �  �  �  �  �  �  |  d  M  =  0  "      L  V  `  i  i  ^  L  :  $    �  �  �  �  �  R    �  y  -    �  H  �    j  �  �  "  8  /    �  4  �  �  �  x  :  L  
     �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   t   f   Y  \  K  9  %    �  �  �  �  �  u  Z  >  "  �  �  �  W    �  @  :  4  ,  "      �  �  �  �  �  �  �  [  �  �  M      �  �  �  �  x  m  `  Q  >  )    �  �  �  �  �  T  !  �  �    ^  K  6    �  �  �  �  [  .  �  �  �  p  @  	  �  g    �  
T  
�  
�  
�  
�  
�  
�  
�  
s  
<  	�  	�  	M  �  ;  �  �  �  �  V  �  �  �  �  �  �  �  �  �  }  d  E    �  �  i  $  �  �  '  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      �  �  �  �  �  �  �  �  }  r  i  a  T  C  3  !    �  �  C  i  �  �  �  	  	  �  �  �  j    �  x    �  P  �  �  J  0      �  �  �  �  �  �  �  �  m  Q  /    �  �  f  &  
�  
  %  '    
  
�  
�  
r  
(  	�  	x  	  _  �  �  
    �  t  d    R  �  �  �  �  �  L  
�  
�  
&  	�  		  s  �  �  �  �  *        �  �  �  �  �  e  C  !    �  �  �  y  M    �  �  G  ;  .      �  �  �  �  i  4  �  �  r    �  p     �   �  �  t  B  :  C  ;       �  �  �  �  c  @    �  �  {  B  
  �  �  �  �  {  p  d  Y  N  D  ;  1  '                 �  �  �  �  �  �  �  ]    
�  
�  
3  	�  	7  �  �  *  �  �  �