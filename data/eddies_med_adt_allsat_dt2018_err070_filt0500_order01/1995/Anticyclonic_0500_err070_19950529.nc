CDF       
      obs    9   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�ě��S�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       Ns   max       P���      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��P   max       =�h      �  t   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�33333   max       @E��
=p�     �   X   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @vp(�\     �  )@   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @N@           t  2(   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�G           �  2�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �C�   max       >-V      �  3�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��b   max       B,N      �  4d   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B,?�      �  5H   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?��P   max       C�[      �  6,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?�!k   max       C�XG      �  7   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?      �  8�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          )      �  9�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       Ns   max       P�      �  :�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�g��	k�   max       ?�e+��a      �  ;�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��P   max       =      �  <h   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��G�{   max       @E�z�G�     �  =L   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G��   max       @vo��Q�     �  F4   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @L�           t  O   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�           �  O�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @�   max         @�      �  Pt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�e+��a   max       ?��t�j     �  QX               
               "   &   O      �   
                                       6      Q   o               =               C         �      -                      $   
         .NsNcd�OE�N�E�N�=%O��AN�t�PK�O-�PLP,��P|�qO)�P}�N�<�N&�O�hO�w}O;uKN	�N��N@O~N���N׸Oc�WO��GP,�O�i�Pn9�P���OtH]Ny�O&'�O�P�PO��N7=2NV�N�:O٭N���N��cO���O��O�!�N�q�O��ON�tO��OL��OL֤O[zN�>�O$~Nu�kO�ͽ�P��/�T���T���D���D����o�D���D���D��;�o;ě�;�`B<#�
<#�
<#�
<49X<�o<�o<�o<�t�<�t�<���<���<��
<�1<�9X<�9X<���<���<���<���<���<�h<��<��<��=C�=\)=\)=�P=��=�w=�w=,1=8Q�=8Q�=H�9=H�9=H�9=P�`=m�h=u=��=��T=ě�=�h��������������������(&))5=BNPNB:5)((((((ggltu�����������thgg����������������������������������������ACIUZahnr������zaUHA���������������������~��������������������������������������nkz���������������wn���)5NZUV@:2)
�����);EB+������jcfdnz����������~znj=;?FN[gt�������t[NB=")*469BHOSOB666)""""BBO[[]ge[OLBBBBBBBBB)5N[\XUP5)dbcht�������������nd��������������������������������������������������������������������������������������������������������������������������������������������"&/;HLMKHC;/#"�����
/7994/'#
 �

�
#2AckliU0#
����������������������)5NXTSWTNB)����55B[���������un[B@C5����������������������������������������USPY[`ehtx�����xth[U|yy|~�������������||������&-
����������!$"���85;<HNTPH<8888888888;<HUaaaa^ZULH<;;;;;;[X[^`hkty���ythc[[[,#%*-;?HMTadkmfaTH;,������

���������?AHITacda`TH????????�������

�����#/<EPOLEDLZUH</#����������������������������������������������)035.)%��$%#�������������������������������������������������� ��������������		�������)56:<;:5)'$�����

����������������������������������)-00+	��溗��������������������������������������Ź����������������ſŽŹűŷŹŹŹŹŹŹ�zÇÓßàâàÜÓÏÇ�z�t�n�m�h�e�n�x�z������������������������������������=�I�V�\�b�o�p�o�b�^�V�I�=�0�/�-�0�5�=�=�����������������������������������������'�3�@�J�K�@�=�3�'���	�����'�'�'�'�N�g���������������s�Z�N�5�%����)�5�N�������
��������������������������������������������������������������������G�T�`�m�~����w�m�T�;����ܾܾ����G�/�H�P�g���������m�T�H�/�$��	�������	�/E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��)�B�[�tāĈą�|�k�O�E�6�������!�)ÇÒÓàáàÖÓÏÇ�|�}�z�y�z�ÇÇÇÇìîñùùùìàÛÚàëìììììììì�����"�*�5�A�<�(����˿ǿ̿ѿӿݿ�������������¾�������������s�l�i�k�r����(�5�N�Z�_�g�j�g�d�Z�N�5�'�������(������������������������<�=�H�J�H�F�?�<�6�/�%�#�� �#�&�/�;�<�<�#�/�<�H�O�H�@�<�8�/�#�!�#�#�#�#�#�#�#�#��"�&�.�9�;�E�A�;�7�.�"���	��	�
��ù��������ùìéàÓÇÆÇËÓàìîùù�����������������������������������������	��"�/�;�F�H�N�T�F�;�/�"��	���������	�`�m�y�����������y�m�`�T�G�;�6�=�>�G�T�`������������ ����ּ�������e�V�X�f��O�\�`�c�V�V�Y�O�C�*�������*�6�C�OƚƳ���
��#������ƧƎ��r�e�^�\�h�uƚ�f�����������Y�����лû����лܻ����f�����������Ŀտ��ݿѿĿ�������������������!�,�*�!���������� �������!�-�:�F�I�S�_�c�m�l�g�_�S�F�A�-�!���!��"�/�;�H�J�N�H�C�;�0�/�"�!����������"�-�0�1�0�/�"�	���������������������G�T�U�\�T�G�;�0�"��	����	���"�.�;�G����������������������������������������ùü������������ùìéìóøùùùùùù����������������������������|�z���������������#�G�I�S�I�D�-�#�����������������#�/�<�B�B�G�<�4�/�(�#�����#�#�#�#�#�n�{ņŇōŌŇ�{�p�n�e�c�n�n�n�n�n�n�n�nD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DzD{D�D�D��f�����������f�Z�M�A�4�(���� �0�Z�f�����1�A�N�B�6�0�)���������ý�������h�tāčēĖĕčā�t�j�h�[�S�[�`�h�h�h�h�#�0�<�I�P�U�X�\�T�I�<�0��	�
�����#����������������������������������������ʼ˼ּ����������ʼ������������Ǽ����
��#�'�&�%�"���
�����������������������������������������y�u�l�e�\�\�`�p���!�-�:�F�N�S�H�:�-�������غ�����!����������������¿²§­²¿������������E�E�E�E�E�E�E�E�E�EuElEkEkEqEuE�E�E�E�E�����	����	���������������������������@�Y�r���������������~�r�f�Y�L�3�$��"�@ _ V % 9 4 < ; 2 8 S C ] = i S b P @ A ` Z X 5 p E 5 # R < N i O > U * - p  l ) F + 6 F y 9 4  ' Y ? d ^ \ _ _ f      �  K  �    \  �  b  t  �  ;  f  ~  �  �  g    �  �  I  �  R  U  �  5  �      /  [  !    �  |  +    o  H  �  �  2  �  �  �  �      	  �    �  �    �  �  �  �C�����<e`B�t��o<o<T��<�`B<t�<�=,1=�1<�h>bN<���<�t�=��=�w='�<�t�<��<ě�<���=�P<���=H�9=@�=���=49X=�
=>C�=@�=�w=8Q�=0 �=�^5=�w='�=�P=,1=��=D��=0 �>-V=u=�E�=�+=�7L=u=��T=�\)=���=ě�=���=��=���>$�/B!�BfsB
D�BSYBS�BܠB ��B��B�oB�eB�vB`B�B	/�B�iB�8B�B�B"FB �dB�B"6B!F�B";9B"�hA��bBJ�B&�B�5B�zB��B>�B 2�B�bB
�3B�yBXB\B{/B�rA���B�=A�E�B(�Br�B�B8�B�B�qB8�Bc�B,NBB�WB�hB��B�]B"5�B|JB
?~Bq�B?\B&B ��BH�B�tBBjB�BX�B4�B	>�B��B?�B�%B
��B@�B �6B��B"@�B!AYB"@B"³A���B9�B%�(B�sB��B�BBgB ?�B �B
��B��B:�B�UB�B�@A��EB�nA�wnB@B��BéB=VB�|B�*BH�B��B,?�B�B8[B�B�!B�@�A�T�A�$8A��!B]�A���?��PA��3B�]A�P�Aa�AA��C�[A�/RAɈ�A�O�A�7�AHS�A��A/�A���A��A_�A�7o@�gHA���Aj�S@�H9B  pB5�@�L]Au�@cB@�wA�L�A��Aa�A�ZA���AH�TA��A�x�A�HnC��kAA$�A�dA��A���A�n
@��\A�^"A@,@i�FA�yuC���A��7?�	@�]A�z�A�yA�� B
��A��?�!kA�a�B�3A�SMAd�UA��C�XGA�}�A��A�z�A��AG��A�xgA/.�A³�A�R�A^��A�|'@�7�A�}sAjEY@�DA�~�BG:@��Av�@d,�@���A���A�yA`��A�|�A�*AH�A�b�A�nA�SC��AA:�A���Aܓ'A�EA�q�A �RA�|�A,@l�A��^C��A�@)[                               "   '   O      �   
   	                                    7      Q   p               =   	            C         �      -            !         %            .                        -      )   +   9      -         #                                 /      3   ?               '                           !   %            '                     %                        )      %   )   '                                                         %                                             %            #                     %NsNcd�Oq�N�E�N�=%O@�NL�P�O-�O�<ZPc<O���N�Z�OK�=Nc:'N&�OS��O�P"O1W�N	�N�QNN@O~N���N׸O6�|ON�Oa��O,7O��O��}O?��Ny�O&'�O�ODH�O��N7=2NV�N�:O- �N���N��cO��O�O�!�N�q�O��ON�tO�\fN�-jN�O?�N�>�OK�Nu�kO���  %  �  o      %  �  ;  �  �  �  U    M  �  <  +  �  �  {  �  -  �  �  *    �  �    �  
�  �  ~  �  ;  �  n  �  A  �  
  %  E  �    9  :  W  �  ]  8  J  �  �  �  �  x��P��/�D���T���D����`B;o;�o�D��;D��<49X=C�<D��=�o<49X<#�
<��
<�C�<�C�<�o<���<�t�<���<��
<��
<���<�/=@�<��=�o=��<�h<���<�h<��=}�<��=C�=\)=\)=��=��=�w=u=D��=8Q�=8Q�=H�9=H�9=L��=ix�=�+=�%=��=��=ě�=��������������������(&))5=BNPNB:5)((((((ihhmtx�����������tii����������������������������������������UMQU[aanxz������znaU������������������������������������������������������������qy����������������vq��)5FONH:62)������%' �������mnnz�����������zqnmmKGHJNP[gt{�����tg[NK#)+567BDOQOB6)######BBO[[]ge[OLBBBBBBBBB$)5BMLHB<5)ecdjt�������������pe�������������������������������������������������������������������������������������������������������������������������������������������� "/2;HIKIE@;/,"�����
$/455/*#
�#)0:<IOW\YUQC<0#��������������������	
)5@BDGHEB5)LIIO[htxz�����xth[RL����������������������������������������USPY[`ehtx�����xth[U|yy|~�������������||��������

���������!$"���85;<HNTPH<8888888888;<HUaaaa^ZULH<;;;;;;[X[^`hkty���ythc[[[84359;CHSTZ^aba_TH;8������

���������?AHITacda`TH????????����

�������"#'/3<AJKHF><2/-)"����������������������������������������������)035.)%��$%#�������������������������������������������������������������������������)56:<;:5)'$������

���������������������������������)-/0*��纗��������������������������������������Ź����������������ſŽŹűŷŹŹŹŹŹŹ�n�zÇÓÞàáàÛÓÍÇ�z�v�n�m�i�f�n�n������������������������������������=�I�V�\�b�o�p�o�b�^�V�I�=�0�/�-�0�5�=�=�����������������������������������������3�8�@�A�@�8�3�'�����'�)�3�3�3�3�3�3�g�s���������������s�g�Z�N�(�!�#�(�5�N�g�������
��������������������������������������������������������������������G�T�`�n�t�w�u�m�`�;�.�����������G�"�;�T�a�o�q�q�l�a�T�H�;�,����������"E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��6�B�O�[�e�h�q�p�f�[�Q�O�B�6�2�.�,�,�1�6ÇÑÓàààÔÓÌÇÀ�~�|ÀÇÇÇÇÇÇìîñùùùìàÛÚàëìììììììì���������%�(�������߿ٿܿݿ�꾌�������������������������y�s�j�l�s����(�5�A�N�Z�^�g�c�Z�N�5�(��������(������������������������/�<�H�I�H�F�>�<�5�/�&�#��!�#�'�/�/�/�/�#�/�<�H�O�H�@�<�8�/�#�!�#�#�#�#�#�#�#�#��"�&�.�9�;�E�A�;�7�.�"���	��	�
��ù��������ùìçàÓÈÌÓàìðùùùù�����������������������������������������	��"�/�;�@�G�H�L�?�;�/�"��	�������	�`�m�y�������������y�m�`�R�G�>�F�G�M�T�`����������ļ�����������������r�m�j�t��6�B�C�O�I�G�C�7�6�*�"������*�3�6�6ƚƧƳ����������������ƳƧƚƓƆƁƃƎƚ�4�Y�f�u�����z�r�Y�@�4�������������4���������Ŀп޿ݿϿĿ�����������������������!�,�*�!���������� �������!�-�:�F�I�S�_�c�m�l�g�_�S�F�A�-�!���!��"�/�;�H�J�N�H�C�;�0�/�"�!�����������	���#�"�!���	������������������G�T�U�\�T�G�;�0�"��	����	���"�.�;�G����������������������������������������ùü������������ùìéìóøùùùùùù����������������������������|�z��������������
���#�'�#�#��
�����������������#�/�<�B�B�G�<�4�/�(�#�����#�#�#�#�#�n�{ņŇōŌŇ�{�p�n�e�c�n�n�n�n�n�n�n�nD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DӾf�s�������������f�Z�R�M�A�<�A�C�M�Z�f�����1�A�N�B�6�0�)���������ý�������h�tāčēĖĕčā�t�j�h�[�S�[�`�h�h�h�h�#�0�<�I�P�U�X�\�T�I�<�0��	�
�����#���������������������������������������弤�ʼּ���� ������ʼ������������������
�������
����������������������y�������������y�u�o�l�h�l�p�y�y�y�y�y�y�!�-�:�>�F�L�Q�F�:�-�!�����������!����������������¿²§­²¿������������E�E�E�E�E�E�E�E�E�E�EuEmElElErEuE�E�E�E�����	����	���������������������������@�Y�r�~�������������~�r�g�Y�L�3�&��$�@ _ V " 9 4 3 6 + 8 Q I ] 6 & ] b = A B ` \ X 5 s E 2 $  ' * \ G > U *   p  l ) 5 + 6 A J 9 4  ' \  h S \ @ _ d      �  >  �    �  i  �  t  '  �  �    �  �  g  �  �  �  I  �  R  U  �  5  �  �  �  ?  (  �  �  �  |  +  �  o  H  �  �  y  �  �    b      	  �  �  �  �  �  �  Y  �  �  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  %  $  #  "  !        
  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  o  n  g  W  >  #    �  �  �  t  @    �  x  #  �  D  �  Q            
       �  �  �  �  �  �  �  �  �  {  i  W              �  �  �  �  �  �  m  N  )    �  �  �  U          !  $  !          �  �  �  �  _  %  �  �  P  �    6  P  f  w  �  �  �  |  p  c  V  B  (    �  g  >    �  �    ,  9  8  *    �  �  x  >    �  �  R    �  }   �  �  �  �  �  }  m  Z  C  )    �  �  �  �  �  �  q  Y  8    G  f  ~  �  �  t  [  <      �  �  �  �  ~  T    �    �  �  �  �  �  �  �  �  �  ~  C  �  �  �  �    >  �  ~  �   �  �    {  �    =  Q  U  O  Q  O  8    �  �    u  �  �  J  �  �  �  �          �  �  �  i  5  �  �  r  C  M  �  8  �  O  �      �  #  B  M  =    �  {  �    �  �  	�  �  :  �  �  �  �  u  b  M  8  $    �  �  �  �  �  �  h  N  D  ;  <  O  b  u  �  �  �  �  �  �  �  �  �  �  �  q  `  N  <  *  �            #  %    �  �  �  �  �  z  C  �  �    �  �  �  �  �  �  �  �  �  �  �  �  a  <    �  �  v     �  g  �  �  �  �  �  �  �  f  C    �  �  d     �  �  G    �    {  x  t  q  m  i  f  b  _  [  Q  ?  .       �   �   �   �   �  �  �  �  �  �  �  |  e  I  )    �  �  ~  Q  '    �  �  O  -      �  �  �  �  �  �  �  
    "  ,  1  7  <  @  D  H  �  �  �  �  �  �  �  �  �    o  _  P  A  3  &     �   �   �  �  �  �  �  �  b  =      �  �  g    �  &  �     �     |  *        �  �  �  �  �  �  �  �    k  W  C  .      �              �  �  �  �  `  ,  �  �  =  �  9  �  �   �  �  �  �  �  �  �  �  �  �  �  Q    �  �  L  �  �  w  L  *    (  7  ?  C  =  -  |  �  �  �  �  Z    �  5  �  �     �  �  �  �    �  �    �  �  �  �  �  n  H  !  �  �  6  �   �  k  �  �    .  [  �  �  �  �  �  �  �  h    �    G  �  W    �  �  	  	-  	�  
  
k  
�  
  
?  
  	�  	e  �  8  �  �  X   �  d  q  {  �  v  h  X  A  '    �  �  �  r  C    �  �  \  Z  ~  {  r  d  U  G  ;  <  5  )      �  �  �  c    �  v  !  �  �  �  �  �  s  `  F  2      �  �  �  �  �  v  [  V  �  ;  2  %      �  �  �  �  �  j  B    �  �  �  �  n  S  7  |  �  �  �  �  �  �  �  �  �  �  �  �  �  `  �  l  �  �  4  n  f  _  S  D  4       �  �  �  �  �  �  �  �  �  �  z  u  �  �  �  �  �  �  �  �  �  �  u  R  3          0  F  [  A  1          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  j  Z  J  9  '    �  �  �  �  �  �  e  E  �  �  	  	x  	�  	�  
  
  
  
  	�  	�  	�  	V  �  )  n  �    �  %  $  #        �  �  �  �  �  |  W  .    �  �  �  v  V  E  =  5  .  &          �  �  �  �  �  �  �  �  �  �  �  V  .  �  �  �  y  :  �  �  N  �  Z  �  �  �  :    d  '  �  �  �  �  �  �      
  �  �  �  �  q  D    �  �  �  h    9  	  �  �  t  @    �  �  L    �  k    �  �  M  �    )  :  :  9  3  &    �  �  �  �  ^  0    �  �  �  u  b  |  �  W  H  7  &       �  �  �  �  �  m  K  #  �  �  �  W  B    �  |  s  k  a  U  I  <  +    	  �  �  �  �  �  G  �  !  c    Y  N  >  "    �  �  �  �  �  s  >  �  �    v  �  T  �  �  �  �      +  8  7  .      �  �  �    G    �  �  D    (  1  3  8  @  F  G  I  I  B  4    �  �  �  �  u  6  �  �  �  �  �  �  �  �  h  0  �  �  y  >    �  �  t  2  �  �  �  p  V  <       �  �  �  �  �  p  _  N  8    -  D  G  G  
  �  �  �    i  N  )  �  �  �  G  �  �    �  >  �    �  �  �  �  y  n  b  V  I  =  1  %        �  �  �  �  �  �  q  s  S  *  �  �  �  �  p  ?    �  p    �  B  �  �  P  