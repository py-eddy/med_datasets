CDF       
      obs    9   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?Ƈ+I�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�o�   max       P���      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       =}�      �  t   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>���R   max       @F�G�{     �   X   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @vq\(�     �  )@   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @P`           t  2(   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�x        max       @���          �  2�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �49X   max       >]/      �  3�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�8	   max       B/��      �  4d   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��n   max       B/�      �  5H   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >��5   max       C�k      �  6,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�Up   max       C�h�      �  7   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          G      �  8�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          )      �  9�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�o�   max       P&�      �  :�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���n   max       ?���rH      �  ;�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       >+      �  <h   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?(�\   max       @F��R     �  =L   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�\(�    max       @vq\(�     �  F4   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @P`           t  O   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�x        max       @�w�          �  O�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         C�   max         C�      �  Pt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?~�Q��   max       ?�s�g�     �  QX                     
      4         n   9                  	      $      %   #               J   5      
               7   !   *            �   '         N            "       +      "   ?   �OePbO��lNN�N�k�P$N��M�o�O��O���O���O� dP���Pg�Ns�kO�N(��Oz��N[�{N�qN�YO��N"�OʠP(Y�O�?N{d�NIu�NE/O��P4��O+h�O;�4O�N{e�OAO�_O��,O�EO��O+FZO�$O"��O�*P�N5O�KwP<aOn�O�hO�O�;kO�JO���N�ROq��O��
Pj����������e`B�o��`B��`B��`B�ě��o��o;o;D��;�o;ě�<o<t�<t�<D��<e`B<e`B<u<�o<�C�<���<���<��
<��
<��
<�1<�9X<�9X<ě�<ě�<�/<�`B<�h=o=+=C�=C�=\)='�='�=0 �=0 �=49X=<j=@�=@�=H�9=H�9=Y�=]/=]/=u=u=}�~y~����������������~LLNRX[gt���������tNL<BBO[c[TOB<<<<<<<<<<xvz������������zxxxx\Z\XZam����������f\dcfgmt������tohgdddd ������������������������")AILKGB5��zx�����������������z
"/HTamqaUJG;/(����)5;D\LL@����O[t������t[G4.2)��������������������
*47=CEA:6*����������������������������������������#/750/#(%$))5@BEFEDB=5)((((#)+-)����� %&$#�����������������������������
#/:72+#
���������/<AD?3#
��������
#/<?HH</#
������������������������������������������[V[bhiptzth[[[[[[[[[gkt�������������tlhg����5BNRJ5)�����������������������BBFN[gt~�����tg[VNHB)0FNPPNFB5)TQRU_anstqneaUTTTTTTnpy���������������tn�����
#051#
������������)10,)(���������
#)-)%!
�������������������������������������������)(+/<HQU`aca[UH<3/))Zaaimvz��������zmbaZ���������
���")58EMSTQMB5) 355BNONNBB>533333333�������*652*�����?=?BOhp����th[OEHJE?::>DHMTadhhgbaUTHE;:� 
#)05:50-%#
����������������������������
��������6BO[_hk_[OHB6)�����������!!������������������������~~����������������������������������{{����$053�����Ŀ����������������������ĿĳĦĝĞħĳĿ�����������������������������������������y�������������y�t�n�y�y�y�y�y�y�y�y�y�yE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��Z�g�������������������������g�N�H�E�N�ZÓàìïìëëàÓÇ�z�s�t�zÇÇÓÓÓÓìù������������ÿùìéìììììììì�������ɺֺ�ֺԺɺƺ��������������������[�h�tĔĢĦĲĦĢčā�t�h�[�W�P�U�Y�Z�[�[�g�n�s�q�t�r�n�g�h�[�N�B�)�&�&�5�G�N�[�;�H�T�H�D�B�F�J�J�C�;�)�"��	����"�;�<�I�S�eŇž����ŠŇ�b�0�
��ļ��������<���ľϾξ��������Z�A�C�@�A�M�f�����������M�Z�_�f�l�f�\�Z�M�F�A�7�A�H�M�M�M�M�M�M�G�T�`�f�m�s�x�y�y�m�d�`�T�G�C�<�9�;�=�G�������������������z�w���������	���"�%�%���	����޾۾׾վ־����	�T�T�a�e�h�a�T�H�F�;�H�T�T�T�T�T�T�T�T�T�ʾ׾��������׾ʾ��������Ⱦʾʾʾ��a�n�x�z�n�g�a�U�L�L�U�Y�a�a�a�a�a�a�a�a�.�;�T�`�s�v�r�f�`�T�G�;�"�	������	��.�ùĹϹܹ��ܹϹ̹Ĺù��ùùùùùùù�������,�2�4�3�%�����������������������������������y�`�G�.����%�5�T�`�y��¦«µ³²©¦�G�F�:�-�(�!�������!�-�:�F�G�G�G�G�m�z���������z�m�i�i�m�m�m�m�m�m�m�m�m�m�F�S�Z�_�`�_�S�M�F�B�E�=�F�F�F�F�F�F�F�F�B�O�Y�b�i�j�i�]�C�6����������)�6�B��(�A�a�g�^�K�8�3�5�(���ٿѿɿӿ������������ĽннĽ½����������z������������������
�������ݽ׽ѽѽٽݽ����G�T�`�m�y���������y�q�m�`�T�;�5�0�3�=�G���)�-�)�'� ���������������	��"�+�/�8�;�<�;�6�/�"���	�	����	�������������������s�f�D�C�A�M�V�f�s�����!�-�:�S�b�o�p�o�v�l�_�F�-���������(�4�A�T�Z�x�����������f�M�A�4�%��#�(������'�,�3�1�'������ܹϹù��ùϹ����ʼռ׼Լʼ�����������������������������������
����������������������������tĀāčđĚĜĞĚĚčā�x�t�g�_�b�h�k�tDoD{D�D�D�D�D�D�D�D�D�D�D�D�D�D{DrDeDcDo����������$�$���������Ƨƃ�h�h�u�|ƚ���=�I�I�U�L�I�=�=�0�0�0�1�=�=�=�=�=�=�=�=�y���������������������y�l�e�`�c�e�d�l�y�������μμ����r�M�@�.������4�M�r����#�0�<�D�I�S�Q�I�<�0�#���
�
�	�
��������!�'�.�7�.�'�!��������������������������������ùêæìòü�����N�Z�����������������s�g�Q�C�,�(�"�(�5�N�r���������κź��������������~�p�a�[�f�r���
���#�5�<�A�<�4�/��
���������������a�n�zÀ�{�z�n�a�V�V�a�a�a�a�a�a�a�a�a�a�(�5�A�N�Q�V�Y�W�V�X�N�;�(� �����!�(���������������������������x�j�g�Z�g�~�����'�4�F�M�?�?�:�'�����ϻɻʻڻ��  > k U b 5 � H ; V c P D + 2 5 & : < P 3 j $ ? o e 8 U  ( ; 5 , J P ! 7 S J I 8 4 ' e 5 X m @ E X M S 0 m 1 I d    �  Z  `    ;  �  b  D  �  �  Y  �    j  D  B  �  l  �  r  �  k  �    �  �  b  >  	    p  �    �  U  �  !  S  O  c  D  h    �  G      a  T  T  8  t  �  8  �    ��t�;D���49X;�o<e`B;D��;o<49X=H�9<u<u=�l�=y�#<49X<e`B<49X<�1<u<�9X<���=L��<��
=Y�=Y�=t�<���<ě�<�9X=��=���=0 �=C�=P�`=o=49X=aG�=�-=�+=���=D��=y�#=�%>]/=��=@�=�C�=��m=�\)=m�h=�O�=��=�1=Ƨ�=ix�=��=��#>Z�BM�B	��B��B�B $�B	�WB�B!��BvrB PA�8	B��BR�B��B/��B��B!�;B�BK�B��B�B F7B�{BW\BxhB��Bc�B�-BB��B %�B	,�BBQlB
��B�B�bB$ABz�B"��B�A��lB�BB��B-��BTA�G�B$�JB.B�$BQB�B�B��BJwB��B>�B	@B�8B=�A��)B
>�B?&B!��B��B!SA��nB>�BBRB��B/�B�bB!��BǍB?=B��B��B I�B�MB@oB�B 0�B�9B��BH�Bj�B�B	hjBCBMqB8�B��B��B$@BD-B"�`B�OB 3�B>�B [B��B-�B�xA�|B$�=BʨB|�B�-B�6B��B�Bz�B��A�ofA�q�A��C�kA�r�A�w�A�@!ʑAܹ�A���A�l�A�MAE�2A=גAgxiAF�\AYc/A���AS�RAƠ�Ab'A>��5A�$Ai��A�|�@p|A�n�@�HA׎�A�,�A"A.-oAhg�A�VA��ACn
@v{A>eO?L�^@��A�+�A�l�C��B��B
�A�>@ߝ�A�'UA	�6A�T�A���@:�A��A�HeA��A� p@���A�YA��A%qC�h�A���A��GA�V�@#UaA܊$A�|�A���A�})AEnA=��AgaZAGJAZ�A��%ATU~A��0AbI>�UpA���Ak	HA��@g��A�o�@�A׈6A���A$"�A.�gAg�A���A��AC�[@uI�A>�?T�@���A�pzA݇-C���B��BA�@�IA�kA|A�ȕA�y@�LA��UA�I�A��zA���@��                     
      4         o   :                  	      $      %   #               J   6                     8   "   +            �   (         N            "       ,      #   ?   �               )                  !   G   5                                 +               #   -                     !                     -      !   +            #   !   !         #   3               )                                                                                                                     %      !   '            #   !            !   N�FN�q$NN�N�k�P&�N��M�o�N�p�Od��Oq��N�WQO���O�lNs�kN�� N(��Oz��N[�{N�#N�YOO�6N"�OoAqOo��N���N{d�NIu�NE/O�$�O��O+h�O;�4OT�N{e�OAO���O��O:�(O2�O+FZN�(_O L:OqUO�%N5O�KwOƀDO4�O�hO�O�;kO�JOn+�N�ROq��O��uO��    �  #  0  D    _  v  Y  E  H  	R    �  �  �  x  �  �  i  �  _     �  �  m  @  �  	Z  >  3  �  U  �  �  �  c  8  �  �    �  �  �     #    n  �    �  �  W  �  �  	�  d����#�
�e`B�o���
��`B��`B��o<49X;o;�`B=}�=C�;ě�<#�
<t�<t�<D��<u<e`B<�/<�o<�`B=+<�9X<��
<��
<��
=,1='�<�9X<ě�<�`B<�/<�`B=+=H�9=#�
=<j=C�=0 �=49X=���=@�=0 �=49X=u=H�9=@�=H�9=H�9=Y�=��=]/=u=�+>+��������������������ZTY[gt�����vtsg[ZZZZ<BBO[c[TOB<<<<<<<<<<xvz������������zxxxx^`[amu�����������mb^dcfgmt������tohgdddd �����������������������)59BFEB=5) ��������������������''/4;HTaga`TOH;/''''����)35775-)��VSRS[ht���������th[V��������������������#*46><64*����������������������������������������#/750/#&%),5>BDDBB5+)&&&&&&#)+-)��������������������������������
#(+*(#
������
&/3960/#
�����
!#/;<C</#
�������������������������������������������[V[bhiptzth[[[[[[[[[pptw��������������tp������.:;90)����������������������BBFN[gt~�����tg[VNHB)5BLMKGCB5)$TQRU_anstqneaUTTTTTTnpy���������������tn�����
#,/2/(#
�������� ������� 	
!#$%#!
��������������������������������������������/-/4<HIUTH?<;///////lcemmyz���������zmll��������

�����%)5GOROKB=5)$355BNONNBB>533333333�������*652*�����ACO[bhp|���th[ONOKCA<;?EHPTabggfaaTH<<<<� 
#)05:50-%#
����������������������������
��������6BO[_hk_[OHB6)��������
������������������������~~������������������������������������������������Ŀ������������������ĿķĳĭĳĵĿĿĿĿ�����������������������������������������y�������������y�t�n�y�y�y�y�y�y�y�y�y�yE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��s�������������������������s�g�M�H�N�Z�sÓàìïìëëàÓÇ�z�s�t�zÇÇÓÓÓÓìù������������ÿùìéìììììììì�������ƺ��������������������������������h�tāčėğĚĐčā�t�h�b�[�W�X�[�b�f�h�N�[�g�m�m�p�o�k�g�_�[�N�5�,�)�0�6�A�K�N�"�/�5�;�?�@�C�A�;�/�/�)�"����"�"�"�"�
��#�0�=�B�C�?�0�&��
���������������
�s���������������������s�f�_�[�Y�\�j�s�M�Z�_�f�l�f�\�Z�M�F�A�7�A�H�M�M�M�M�M�M�T�]�`�k�m�o�m�l�`�T�J�G�A�>�G�J�T�T�T�T�������������������z�w���������	���"�%�%���	����޾۾׾վ־����	�T�T�a�e�h�a�T�H�F�;�H�T�T�T�T�T�T�T�T�T�׾��������׾ʾǾ¾ʾ̾׾׾׾׾׾��a�n�x�z�n�g�a�U�L�L�U�Y�a�a�a�a�a�a�a�a�;�G�T�`�d�`�T�L�G�;�.�"���	��	��.�;�ùĹϹܹ��ܹϹ̹Ĺù��ùùùùùùù�������"�)�+�)�$��������������������m�y������������y�`�G�;�7�4�;�G�U�`�f�m¦©²±¦£¡�G�F�:�-�(�!�������!�-�:�F�G�G�G�G�m�z���������z�m�i�i�m�m�m�m�m�m�m�m�m�m�F�S�Z�_�`�_�S�M�F�B�E�=�F�F�F�F�F�F�F�F�6�B�J�O�W�^�_�]�O�B�6�)�������+�6���(�3�A�E�M�F�A�(����������������������ĽннĽ½����������z������������������
�������ݽ׽ѽѽٽݽ����G�T�`�m�y������w�m�`�T�G�;�:�4�7�;�B�G���)�-�)�'� ���������������	��"�+�/�8�;�<�;�6�/�"���	�	����	�f�s����������������s�l�Z�P�M�H�E�M�Z�f��!�-�:�W�b�c�_�S�F�:�!���������A�Z�n�s�������z�s�f�Z�M�A�9�,�&�(�4�A������ �'�,�*�'�������ܹҹѹܹ�����ʼռ׼Լʼ�����������������������������������
�����������������������������h�tāččĚĚĜĚĕčā�|�t�j�h�b�g�h�hD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DwDvD{D�D�ƚƧ������������������ƧƉƁ�p�n�uƆƚ�=�I�I�U�L�I�=�=�0�0�0�1�=�=�=�=�=�=�=�=�y���������������������y�l�e�`�c�e�d�l�y���������������r�M�@�4�$���&�@�M�f������#�0�<�A�I�O�L�I�<�0�#����
����������!�'�.�7�.�'�!��������������������������������ùêæìòü�����N�Z�����������������s�g�Q�C�,�(�"�(�5�N�r���������κź��������������~�p�a�[�f�r�����
��!�*�&�#����
�����������������a�n�zÀ�{�z�n�a�V�V�a�a�a�a�a�a�a�a�a�a�(�5�A�N�Q�V�Y�W�V�X�N�;�(� �����!�(�����������������������������{�p�g�s�������'�+�-�-�+�'�������������  ; k U d 5 � ) / O `   + 6 5 & : 0 P 2 j * H u e 8 U  % ; 5   J P  1 J D I ( 1 % a 5 X l > E X M S , m 1 > F    �  
  `    �  �  b  �  �    B  H    j  �  B  �  l  �  r  �  k  �    [  �  b  >    �  p  �  �  �  U  )    �  �  c  �    �  g  G    H  =  T  T  8  t  �  8  �  �  Z  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  �  �  �  �  �    	      
  �  �  �  �  w  <  �  �  Y   �  �  �  �  �  �  �  �  �  �  �  �  �  �  k  L  &  �  q  �  u  #                               �  �  �  �  �  0  %            �  �  �  �  �  �  �  �  �  �  L  
  �  9  D  C  =  :  0       �  �  �  �  �  �  [  %  �  �  T   �        �  �  �  �  �  �  |  a  A    �  �  �  v  K  #  �  _  M  :  V  x  �  +  C  8  *      �  �  �  �  �  �  x  c  �  �  �  3  ^  o  u  q  f  V  =  !    �  �  �  �  f  ?  �  �    B  P  W  Y  S  B  $  �  �  |  0  �  l  �  C  ]  '  �    7  B  D  D  B  ?  :  ?  :  *    �  �  �  c  -  �  �  l  �    %  /  9  A  G  H  A  6  *      	  �  �  �  �  �  r  �  0  �    *  '    �  	  	C  	R  	E  	  �  (  �  �  �  T  �  5  �  �    B  x  �  �  �        �  �  {    �  �  5  �  �  �  �  �    s  g  [  L  :  '    �  �  �  �  n  J  &    e  l  s  z  ~  �  �  �    z  s  i  `  W  N  F  <  ,      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  {  x  v  x  u  q  j  a  X  N  D  9  -      �  �  �  �  �  r  Y  E  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  o  W  @  (    �  �  �  �  �  �  �  �  �  �  x  \  ?    �  �  �  �  o  J  i  \  O  B  6  1  ,  '  "            �  �  �  �  �  �  +  I  a  v  �  �  �  �  �  �  o  S  0    �  �  <  �  �    _  \  X  U  Q  M  E  >  6  /  &          �  �  �  a  7  |  �  �  �            �  �  �  �  X  "  �  �    T  =  v  J  ;  r  �  �  �  �  �  �  l  @     �  ;  �  �  G  �  *  �  �  �  �  �  �  �  �  �  �  �  '  �  �  �  �  _  5    �  m  j  g  d  [  R  H  ?  6  -  $              p  �    @  :  4  /  )  %  &  (  )  *  1  ;  F  Q  \  a  c  f  h  k  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  o  �  	
  	3  	J  	X  	U  	C  	"  �  �  �  5  �  j  �  b  �  3  I  q  �  �    '  6  =  =  7  "    �  �  ^    �  ?  �  �  �  3  '        �  �  �  �  �  n  J  %  �  �  �  d  >    �  �  �  �  �  q  `  N  =  -      �  �  �  �  �  �  y  J    2  E  P  T  P  G  8  $  
  �  �  �  k  /  �  �    �  1  �  �  �  �  �  �  |  n  a  T  H  ;  /      �  �  �  �  �  �  �  �  �  ~  _  ;    �  �  �  Z  "  �  �  u  4  	  �  �  X  q  �  �  �  �  �  t  b  G  )    �  �  �  [    �  q    �  �      V  ^  b  c  \  M  ,  �  �  R  �  �  &  �  �    W  �    (  3  8  1    
  �  �  �  �  f  4  �  |  �  6  N   K  �  	  9  Z  r  �  �  {  f  7  �  �  c    �  `    �  C  �  �  r  \  E  5  )        �  �  �  �  k  X  `  ?  !    �  q  �  �  �  �        �  �  �  �  G    �  z  3  �  �    �  �  �  �  �  �  �  �  �  }  T  (  �  �  �  =  �  �  m  �  �  >  �  %  f  �  �  �  �  T    j  �  �  �  �  x    -  	
  �  �  �  �  �  �  �  �  �  �  �  h  -  �  �  1  �  }    �                       �  �  �  �  �  �  �  �  �  �  #        �  �  �  �  �  _  &  �  �  w  �  �  ~  %  �  U  e  �  �      �  �  h  
  
�  
$  	�  		  �  '  G    �  �    _  j  m  j  c  S  >  &  	  �  �  w  ;  �  �  e    �  _  9  �  �  �  �  �  �  �  �  �  |  t  `  D  -    �  �  �  y  J    �  �  �  �  �  �  �  �  �  �  i  :  �  �  L  �  �  Z  �  �  v  _  D  #    �  �  �  S  #  �  �  �  \    �  c  �  '  �  �  �  �  �  �  �  �  y  D    �  �  \  ;  �  |  7  &  ;  �    5  J  T  W  T  H  2    �  �  Y  �    �  5  �  �  #  �  �  �  �  �  �  �  �  �  �  w  c  P  0  �  �    D  
  �  �  �  k  <    �  �  �  L  	  �  �  .  �  }    �    �  �  	�  	�  	�  	�  	�  	y  	?  	  �  �  Q    �  V  �  -  i  �  �  �    �  S  �  �      '  N  a  A    �  �  1  e  
l  �  �  n