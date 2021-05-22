CDF       
      obs    8   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��;dZ�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�^�   max       P��J      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��`B   max       =�/      �  l   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�(�\   max       @Fffffg     �   L   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @v�          �  )   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @P�           p  1�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�I�          �  2<   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ����   max       >�        �  3   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��d   max       B.z      �  3�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       B =�   max       B-�]      �  4�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?��   max       C��2      �  5�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?�   max       C��c      �  6�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7|   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          5      �  8\   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          /      �  9<   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�^�   max       P2B      �  :   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�+j��f�   max       ?���"��a      �  :�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��`B   max       >�      �  ;�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�(�\   max       @Fffffg     �  <�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @v~�\(��     �  E|   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @P�           p  N<   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�3`          �  N�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         B�   max         B�      �  O�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���Z�   max       ?��	k��~     �  Pl                     �            	               ;   6                  ,   0                  D      G   )   !   �      !   C                     
               �      
      	      N�:O3�N���O��OA��N�IP��JO(�M�^�N���NĂ�N�p:O͕�O#[�Of��O���O�|6N��/O?jDN�0N8-N�dO�WPM�O���N@m�O_>N�aN�fP��Nw[�O��MP1P\O��hPK53O�D8P_P?��O>�O��N��!O�R�N�0;N��N�Q�O�N��mN:M|NB4�O�T�N��N��ZOB��N��OnyOh��`B��t��49X�o%@  %@  :�o:�o:�o;o;��
;�`B<o<D��<D��<�o<�o<�C�<���<�1<�1<�9X<�9X<���<���<�/<�`B<�`B<�=o=o=+=\)=\)=\)=t�=t�=�P=��=,1=H�9=L��=]/=]/=aG�=e`B=m�h=q��=y�#=�+=�\)=��=���=�-=�
==�/�����
�����������NQX[gt�������|tg[XNN��������������������Y[knt������������t\Y�������
������++-/3<DHPUYWUOH<3/++$B[g�������gNB%����
#050+))&#
��38<<IIII@<3333333333"))146BD<=?:60)behmort{��������tphbrruxyzz��������zrrrr��5BHHNFFL;)�����������������������edght�����������tige7015<HUapxyyxnkaUG<7����������������������������������������ikmpst������������ti
#-/01/##

##0640'###########edht����������|tkhee���0<@CCA<0
��Oht�������th[O@6)��������������������*%&0<=A=<0**********�����������������������������������������������������������ty���������������ztuuz�������}zuuuuuuuu����������������� &)5=FQSJ5��tt{������������������������
���������������������������)5Ngt����t[B5�������)KQQJB6)����������������������������������fhhpt�������xtqhffff������25.$����

#%),/0/#


E?BEHRUY\`afeaaUNHEE)6=8641)aURH><549<HUVabhgaaa:50<DHOUWYXUPHD<::::)))(!%�������
!
�����UPV[_htuttojhhb[UUUU�����


����������������#&%���4356BEJNPUVONKB;8544����
!"
������������

������׽нݽ����ݽнʽ˽ннннннннн��V�a�n�p�|Á��z�n�h�a�a�Y�U�N�J�G�H�T�V������"�)�6�6�<�6�-�)��������������������ɾѾ۾ʾ����������s�i�g�p�s������)�5�B�F�C�5�)����������������������
�����������������������������6�O�[�h�yĀ�~�r�[�6����������������S�_�l�z�������x�s�l�_�S�F�B�9�:�F�M�S�����������������������������������������������������ĿѿԿؿѿĿ����������������-�:�<�F�S�_�d�l�p�l�_�S�F�C�:�8�0�-�*�-�uƁƎƚƧƳƶƺƳƧƚƎƁ�u�m�n�u�u�u�u�����	��	�����������������������������������)�,�+�2�)�#����������������N�[�a�g�l�s�t�o�n�g�[�N�B�=�<�8�7�B�E�NFF1F=FJFJFPFPFOFJF=F1F$FFE�E�E�E�FF�����'�6�>�>�4�'����ܻ������ûܻ���ÇÓÚÕÓÇÄ�~�z�n�b�a�Z�a�f�n�r�zÆÇ�"�/�;�H�T�V�_�_�V�T�H�A�;�/�+�(�!���"���(�)�+�0�(�$������ �������ʼּ�����ּͼʼȼʼʼʼʼʼʼʼʼʼʽĽнܽܽ׽нͽƽĽ������������������ĽĽݾ�(�4�M�Z�_�g�d�Z�4������ݽӽƽнݾؾ޾־;ɾ�������f�R�H�B�R�s�����žʾؾ����"�)�.�4�3�4�.�"��	������ݾоо���������������|�u�����������ѿտݿ�����
�������ݿԿѿʿƿȿѻ!�-�:�>�F�S�[�_�l�s�l�_�F�@�:�-�$�!��!�ܹ������������ܹййܹܹܹܹܹܹܹ�āĚĦĶ������������ĳĦěĐć�l�e�h�sā�(�5�9�>�>�5�(�#�����(�(�(�(�(�(�(�(�����ʼͼ������ʼ������������������6�O�hƎƧ������������ƳƎ�u�\�B�&� �&�6��������������������������������������������ûл�����ܻлû��x�S�=�F�_�{�����(�5�N�Z�s���������~�s�l�N�5������(���� ��'�*�*�"���ݿѿĿÿ̿ʿ̿ݿ���/�;�H�O�R�G�6�"��	������������������/��(�4�;�A�I�M�R�O�M�A�4�(������������������������ֺɺȺɺպ�L�U�Y�e�m�q�m�e�Y�P�L�B�@�?�@�H�L�L�L�L�`�l�y�������������y�l�`�S�O�H�D�C�G�S�`�`�m�q�q�m�d�`�T�G�@�;�7�/�6�;�G�T�_�`�`���������������������������s�s�m�s�{�����~���������������������~�y�}�~�~�~�~�~�~��������(�4�<�=�6�4�(�(�������"�/�6�/�,�"���	��������	������������ ����������������������������������#�%�,�&�#��������������D�D�D�D�D�D�D�D�D�D�D�D�D{DoDfDhDrD{D�DƼY�f�r�|�������s�r�f�Y�M�K�M�N�Y�Y�Y�Y�{ǈǔǡǬǨǡǔǈ�{�{�s�{�{�{�{�{�{�{�{¿����������������������¿²±²¸½¾¿�o�{ǆǈǍǈ�{�s�o�b�V�I�I�I�K�V�b�i�o�o�n�r�{łŇňŇŇ�{�n�b�U�I�I�H�I�U�b�l�nE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E|EvEzE�E� N I e 0 I   3 h � \ n 2 P B  : � , ) * T Q S ) , ^ � c ] : \ q X 8 X N O $ T 7 8 d E - ; r S @ @ N 8 0 a " ?  (  o    2  �    ~  q      �  3  �  �  �  7      �  �  5  7  U  $  F  F  ^    �  
  �  _      �  &    �  �  D  �  �  �  �  �  *  �  Z  f  M  �  �  �  �  8  5����;D��;�o<49X<T��<�9X>�  <D��;D��<o<e`B<e`B<���<�h<�`B=��P=�\)<�1=�P=+<���<�=�+=��=aG�<��='�=\)=t�=���=�w=���=���=�C�>+=��=�O�=���=e`B=aG�=�t�=��-=}�=y�#=�o=�O�=�%=y�#=�+>s�F=���=��=��=ě�>J>O�BB	�B��B
��B4"B��B�UB$�B&JSBu�B~XA��dB3�B'�B
�?B��B!�B�mB
}NBq�B%s9B��B%F�B�SB I�B%��BoB*��B��B��B��BvB=B��BDBڭB	�B�RB��B��B�4B.zBq�B��B.qBs\B=|BPB�HB1#B��Bu�Bc�BߛB�B��B�bB	��B�KB
�B?�B�>B�B$��B&G B?�B��B =�BB�B��B=�B�B!�B�VB
�oB�OB%^ B�kB%<�B�CB ^|B%��B�yB*�B�yB�>B�/B��B��B6�BzB�*B�]BA�B�wB�B��B-�]B@B��B7B?�B@3B��B��B>JB�B�iB��B�xB=�BƸA*�yAƔ|A�wAIX^A�A�A�T&A֕q@��k@��6Au��@���B�A�`&A�:A�!C��2@���AȐ�A�� A�.�A��A&{jA5��AH�AZ�y@�dA~�@���?��A�&A���@���B�<A��;@��A��A�iTA���A7�@OO�?���ADzAf�A���@��A5|A���A���A�ٸC���@��B�A��B�JA�oC��A)x�AƘ�A֑�AGC3A��7A�~�Aֆ�@�&�@���AsI�@�ؓB�*A��AՀ:A���C��c@�Aȡ�A���A��.A�A&�A5��AG<AZ��@�2A*i@� ?�A߀�A���@��AB�A��1@�u�A���A���A���A9�@S�?��A��Ah�A��u@Z�A6��A��\A���A�dC��@�O@B2vA��BB0A��aC��                     �            
               ;   7               	   -   0                  E      H   *   "   �      "   D               	      
               �      
      
                  '         5                              %                  '   /                  )      %   1   #   1   #   -   /                                    !                                       !                                                                     !         /   #   #      #   /                                                      N�:O	YN��Ȍ�O-�N�8�O�bO(�M�^�N���N���N�p:O^"�O#[�ONFO��O3�(N��/N���N�0N8-NҥO��$O�ԢO��VN@m�N�TTN�aN�fO��Nw[�O[S�P*�eO��hO��NO�fO�·P2BO>�O��NN=�O�R�N�0;N��N�Q�N���N��mN:M|NB4�O���N��N��ZO.ԀN��OnyN��^  e  b  D  c  �  M  {  �  �    �  �  c  \  �  	  r  �  �  9      �  M  �  E  �  8  s  	�  �  
M  I  ,  p  v  �  N  L  �  #    s  �  �  �  /  j  �  *    �  �  �  N  	���`B��C��t�%@  :�o;ě�>�:�o:�o;o;ě�;�`B<e`B<D��<T��=��=�P<�C�<�/<�1<�1<�j=t�=#�
<�h<�/<�<�`B<�=e`B=o=e`B=t�=\)=��w='�='�=#�
=��=,1=e`B=L��=]/=]/=aG�=ix�=m�h=q��=y�#=���=�\)=��=��w=�-=�
==�;d�����
�����������ZORY[fgt�����{tg\[ZZ��������������������]^ot�������������tg]�������

������/./2<HKSOH?<:///////2/05BN[gsz���}tg[N82����
#050+))&#
��38<<IIII@<3333333333"))146BD<=?:60)jnpst}��������tjjjjrruxyzz��������zrrrr��)35:<5))�����������������������nhhkt������������tnn;:<?HUainmiaZUPH@<;;����������������������������������������sttw�����������tssss
#-/01/##

##0640'###########ffht�����������tmhff	#07<>><:0#GDGO[ht�������th[WOG��������������������*%&0<=A=<0**********�������������������������������������������������������������������������������uuz�������}zuuuuuuuu����������	������� )5<BEPSJ5��tt{����������������������������������������������������!)5Ngo{�}t[B5$�����)FNOHB6)����������������������������������lltu�������tllllllll������25.$����

#%),/0/#


E?BEHRUY\`afeaaUNHEE)6=8641)65:<HSU`agfaUH@<6666:50<DHOUWYXUPHD<::::)))(!%�������

�����UPV[_htuttojhhb[UUUU�����


���������������"%$����4356BEJNPUVONKB;8544����
!"
�����������

�������սнݽ����ݽнʽ˽ннннннннн��H�U�a�n�o�z�{À�~�z�n�a�Z�U�O�K�H�H�H�H�����(�)�-�6�;�6�,�)������ ���������¾ʾɾʾ�����������s�k�j�t�x������)�5�B�D�A�5�)�����������������������������������������������������)�6�B�P�Z�_�_�[�O�B�)�������������)�S�_�l�z�������x�s�l�_�S�F�B�9�:�F�M�S�����������������������������������������������������ĿѿԿؿѿĿ����������������:�F�S�_�a�l�o�l�_�S�J�F�E�:�:�3�:�:�:�:�uƁƎƚƧƳƶƺƳƧƚƎƁ�u�m�n�u�u�u�u������������ ����������������������������������)�,�+�2�)�#����������������B�N�[�g�k�r�s�n�m�g�[�N�B�>�=�;�9�<�B�BF$F1F6F=FFFGFDF=F1F$FFFF	FFFF!F$F$������%�'�0�/�'�������������ÇÓÚÕÓÇÄ�~�z�n�b�a�Z�a�f�n�r�zÆÇ�/�;�<�H�R�T�T�T�K�H�;�6�/�,�&�,�/�/�/�/���(�)�+�0�(�$������ �������ʼּ�����ּͼʼȼʼʼʼʼʼʼʼʼʼʽĽнֽؽսн˽ĽĽ������������������Ľľ���(�4�A�L�T�T�M�D�4��	���������������ǾȾ�������������q�`�Z�W�j�s�������	��%�.�1�0�1�*�"��	������վ׾�����������������|�u�����������ѿݿ��������������ݿ׿ѿͿͿѿѻ!�-�:�>�F�S�[�_�l�s�l�_�F�@�:�-�$�!��!�ܹ������������ܹййܹܹܹܹܹܹܹ�čĚĦĶ��������ĿĳĦĚĒčā�y�t�vāč�(�5�9�>�>�5�(�#�����(�(�(�(�(�(�(�(���ʼּݼ������ּʼ����������������6�O�hƎƧ������������ƳƎ�u�\�C�(�"�(�6��������������������������������������������ûлܻ߻ݻ׻û������y�w�y�~���������5�A�N�Z�i�t�u�g�N�A�5�(������(�.�5�������"�%�%����ݿѿ̿ѿοҿݿ���/�;�H�P�E�4�"��	��������������������/��(�4�;�A�I�M�R�O�M�A�4�(������������������������ֺɺȺɺպ�L�Y�e�e�k�e�]�Y�X�L�G�E�L�L�L�L�L�L�L�L�`�l�y�������������y�l�`�S�O�H�D�C�G�S�`�`�m�q�q�m�d�`�T�G�@�;�7�/�6�;�G�T�_�`�`���������������������������s�s�m�s�{�����~���������������������~�y�}�~�~�~�~�~�~��(�4�9�<�4�3�(�%�������������"�/�6�/�,�"���	��������	������������ ����������������������������������#�%�,�&�#��������������D�D�D�D�D�D�D�D�D�D�D�D�D�D{DpDrD|D�D�D��Y�f�r�|�������s�r�f�Y�M�K�M�N�Y�Y�Y�Y�{ǈǔǡǬǨǡǔǈ�{�{�s�{�{�{�{�{�{�{�{������������������������¿³³¹¾¿�����o�{ǆǈǍǈ�{�s�o�b�V�I�I�I�K�V�b�i�o�o�n�r�{łŇňŇŇ�{�n�b�U�I�I�H�I�U�b�l�nE�E�E�E�E�E�E�E�E�E�E�E�E�E�E|EwE�E�E�E� N L Y / H & & 3 h � M n  P @   � 3 ) * U 6 ; " , U � c [ : > m X  ; A N $ T 9 8 d E - 9 r S @ > N 8 + a " >  (  A  �  �  �  �  �  q      �  3  �  �  �  6  w    �  �  5    �  �    F      �  f  �  �  �    �  )  =  �  �  D  q  �  �  �  �    �  Z  f  e  �  �  z  �  8  &  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  e  ^  W  P  I  B  ;  3  +  #            +  <  L  \  l  D  a  T  E  3      �  �  �  �  �  m  C    �  �  �  �  �  �    1  \  �  z  ,  r  z  �  �  �  �  �  �  �  Q  �  �    O  Y  `  b  \  S  G  9  $    �  �  �  �  �  �  {  M     �  �  �  �  �  �  �  �  �  p  W  =  !    �  �  �  �  s  F  	  �    ,  >  I  M  G  1    �  �  �  j  4  �  �  {  A    �  �  �  :  �  �  �  T  �  4  h  z  L  �  �  �  k  [  �  	%  �  �  �  �  �  �  �  �  �  �  �  |  r  h  ]  R  F  9  *      �  �  �  �  �  u  i  ]  Q  E  9  -  "          �   �   �   �        �  �  �  �  �  �  �  �  �  �  �  �  �  �  H     �  �  �  �  �  �  �  �  �  w  r  t  z  �  �  �  �  �  �  �  �  �  �  �  �  �  s  c  Q  @  3  '      �  �  �  �  �  �  z  �       (  H  ^  c  b  ^  X  L  ;     �  �  �  �  G  �  �  \  R  P  =    �  �  �  n  ?    �  �  �  W  "  �  �  �  T  �  �  �  y  l  _  R  H  ;  *    �  �  �  C  �  �  -   �   �  �  �  2  s  �  �  �  	  	
  �  �  �  l  %  �  -  �  �  �  �  s  �  �  �  �  e  n  q  p  c  J  #  �  �  e  &  �  4  �    �  �  �  �  y  j  a  X  O  E  9  (      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  j  T  <  %       �  �  �  9  5  -  "    �  �  �  �  �  t  R  4    �  �  �  �    1    �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  X  5    �  	               �  �  �  �  �  �  h  E  #     �   �   �  H  T  y  �  �  �  �  �  �  �  q  >    �  x    �  �  �   �  u  �    '  ?  K  L  F  =  0     	  �  �  c    �  ,  t  8  �  �  �  �  �  �  �  �  {  _  >    �  �  f  *  �  �  8  _  E  F  G  I  J  J  H  F  E  C  ?  :  4  /  *  $          �  �  �  �  �  �  �  �  i  N  *    �  �  �  �  o  ^  a  z  8  '      �  �  �  �  �  �  �  �  p  U  3  	  �  �  s  ;  s  k  c  Z  P  E  :  0  &      �  �  �  �  �  u  R  .    |  �  	!  	O  	{  	�  	�  	�  	�  	�  	N  		  �    j  �  �      �  �  �  �  �  �  �  y  l  _  O  @  0  "      �  �  �  �  �  �  	�  	�  
  
+  
A  
L  
H  
6  
  	�  	v  	  �    �  �  ;  E  �  ,  I  G  <  +    �  �  �  s  )  �  �  G  �  �  �  �  F  v  ,  *  $        �  �  �  �  t  J  "  �  �  h  N  �  �  �  
�  y  �  "  S  i  p  d  ?    �  �  F  
�  
I  	�  �  `  �  5      h  s  u  k  ]  C    �  �  P     �  h    �  e  �  �  x  �  �  �  �  �  t  Z  :    �  �  �  �  S    �  b    �  G  N  D  *    �  �  �  �  �  �  �  f  &  �  Z  �     �  :  L  G  J  J  F  >  -    �  �  �  �  �  �  x  a  ?    �  B  �  �  �  �  �  �  v  a  P  >  *    �  �  �  �  �  l  2  �  �             #  !      �  �  �  ^     �  �  H    �    	  �  �  �  �  c  ;    �  �  �  u  5  �  �  z    �  �  s  a  N  8    	  �  �  �  �  �  �  �  X     �  y  *   �   �  �  �  �  �  �  �  �  �  �  �  u  ^  @  "    �  �  �  d  5  �  �  �  �  �  �  �  �  �  �  t  h  Z  M  @  4  )          �  �  ~  y  r  i  Y  G  5  #    �  �  �  �  �  |  Z  H  /  &          �  �  �  �  �  �  �  �  �  �  k  R  9    j  m  o  q  t  v  y  {  }  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  v  n  a  U  I  <  /  "      �  �  �  �  ~  �    )  &    �  �  c  �  L  �  �  �  =  B  �  	x  x        �  �  �  �  �  �  j  H  '    �  �  �  z  �  q  @  �  �  �  �  s  `  N  ;  )      �  �  �  �  �  �  �  �  �  �  �  �  �  �  h  H  &  �  �  �  p  3  �  d  �  G  �    p  �  �  �  g  J  '    �  �  �  \  0     �  �  J  �  �  <   �  N  *  �  �  �  U     �  �  �  [  :    �  �  �  �  �  �  �  	�  	�  	�  	�  	{  	J  	  �  �  7  �  }    �  P  �  |    �  T