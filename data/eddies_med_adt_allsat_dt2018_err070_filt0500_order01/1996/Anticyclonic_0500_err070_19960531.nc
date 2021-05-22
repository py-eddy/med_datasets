CDF       
      obs    <   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�~��"��      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N�<   max       P���      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���
   max       =�7L      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?p��
=q   max       @E��
=p�     	`   |   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ����P    max       @vrfffff     	`  )�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @P@           x  3<   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ˊ        max       @��           �  3�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ;o   max       >���      �  4�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�c�   max       B,�l      �  5�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B,ö      �  6�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >���   max       C�q      �  7t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�*{   max       C�x      �  8d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  9T   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          3      �  :D   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          -      �  ;4   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N�<   max       P�      �  <$   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Z���ݘ   max       ?�jOv`      �  =   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���
   max       >	7L      �  >   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?���R   max       @E�z�G�     	`  >�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�33334    max       @vrz�G�     	`  HT   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @P@           x  Q�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ˊ        max       @�K�          �  R,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         B8   max         B8      �  S   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?}��,<�   max       ?�hr� Ĝ     �  T                  #                              T         9            
            8   
   "      !   	         !         	      ;      ;      	         -   $         �               .   *         -O�>#OWR9N�&N��<N�IOΓ�N���N��P.b�N�<P!P�NJ�}O�`mNR�N��P���N�|�N���PM��N+bOT�O�N�d�ON:��N�BP s�N�fO��N���PBzNո�N�R�N�\,P6�\O~�.N3�sN|�iNB�%P,��N
Y�P��Orr�N�OaXPN��O���P	O��N�9,P��N-��O%{O�NFXcO;r�O���N���Nn�O�Ż��
���
�D����o$�  :�o:�o;o;�o;��
;��
<o<o<t�<t�<#�
<#�
<#�
<49X<49X<49X<D��<D��<D��<T��<e`B<�t�<���<���<�9X<�9X<�9X<�j<���<�h<�h<�h<�<��=+=��=��=�w=�w=#�
=#�
=0 �=@�=H�9=P�`=aG�=e`B=ix�=ix�=�o=�o=�+=�+=�7L=�7L$$&")5N[guolrpiNB5)$IFDN[gpt����tg[XONI;7<>=<7<IUV``UMIA<;;+()//<HQU`\UQH</++++���	 ����������������������������������������������������������������������
'#/:/
��������������������������dft���������������td" b_aht|�����������thb/565+05BCJFB:5//////�������������������������)Ngpn]?5)����NIFIOW[hltvtrhf[TONNrsntu������������{tr��������
/88-#
����`\adknnzynfa````````��������������������"#&/7<>?><</#^Zanz����zna^^^^^^^^qtutt~������������tq�������������������� )6BDDB6.)          ���6BHMNKH@6)��@BDNN[ghnmkg^[NJDB@@(%'(./<HUbmma^[UH</(��������������������������
 ,<QXS<#
���
#0;<610###�������the`ejqtw����SNU[gty{tsg[SSSSSSSS�BL]d[Y_`[B)��srht�����������{yyus�����������������������

����������������������������������)9>>7)�����������

�����������6BJKB<6.)������).5BCKB@5�Z[gt����tg][ZZZZZZZZ����������������������������������HJ^mz��������zm\TLHH ��)25BNSWWOB7) ��������������������
%)069ACB6)�������
!')'"
����@DHU^afaUH@@@@@@@@@@���������	

������#)/38;=9/#���������������������������������������qs{���������������vq����������������������	


 ��������������������������.�;�G�`�t�|�v�u�{�y�m�`�T�G�;�'�"�#�(�.�������������������������������������������������������������������������}��E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�ù��������ùìæìóùùùùùùùùùù�ܹ��������������Ϲù������ιܼ��������������������������������������������������������������������������������[�o�s�x�j�r�s�g�A�%����������4�B�[��������������������������������������򾱾��̾;Ǿ���������m�j�q�h�f�Z�f������������
���������������������������������%�4�(�������ݽĽ������Ľڽ����������������������������ý�������������I�V�b�l�i�b�V�I�F�G�I�I�I�I�I�I�I�I�I�I�����(�;�=�8�.�(���Ŀ��������̿տݿ꼋�������������ü������������������������3�@�L�X�Y�_�e�l�r�r�e�Y�T�N�L�@�6�3�2�3��(�5�N�Z�g�������g�D�������ֿ׿���a�n�w�zÇÇÇ�z�n�d�a�a�a�a�a�a�a�a�a�a�~�������������������������������~�|�y�~��������������������������s�p�f�l�s����/�<�E�A�?�<�<�/�-�%�#�(�/�/�/�/�/�/�/�/�#�+�/�:�<�H�J�P�N�H�B�<�3�/�%�#���!�#ìù����������ýùöìäìììììììì�'�3�,�*�'������'�'�'�'�'�'�'�'�'�'�����	�"�4�?�N�R�H�;�"�	�����������������U�Z�b�n�n�q�n�n�b�U�I�B�<�9�8�<�I�P�U�U����������������������������������6�@�A�=�9�6�)������)�,�6�6�6�6�6�6�"�/�V�a�j�k�g�`�T�;�*��	�����������"���������������ּּּڼ����F�:�9�6�-�!��!�,�-�:�F�S�_�`�_�[�S�H�F�Ľнݽ���ݽнĽĽ����ĽĽĽĽĽĽĽ���A�M�S�L�H�/��	�������������������	��������������������g�Z�X�V�Z�g�����������;�G�T�`�m�t�m�`�T�G�;�5�;�;�;�;�;�;�;�;�U�a�l�n�o�n�h�a�]�U�H�<�6�<�H�S�U�U�U�UE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��#�<�I�U�[�]�[�U�K�=�#�
������Ŀ�������#E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��~�����κܺۺӺԺպ������~�o�e�U�_�e�r�~�tāčĖĚĢģĘčă�u�t�h�^�[�W�S�[�k�t�h�h�k�s�p�h�[�U�P�R�[�g�h�h�h�h�h�h�h�h�H�U�a�j�i�b�a�a�f�a�U�J�H�<�2�/�-�0�<�H�������������������{�y�p�m�m�y�~��������ĜĦĳķĸĵīĦĚčā�|�s�r�uāĈčĘĜƧ�������2�=�B�J�H�0��������ƳƖƠƧ�M�Z�f�g�h�f�a�Z�P�M�J�A�4�-�,�3�4�A�I�M�-�:�F�J�:�-�(�!��������������#�-DoD{D�D�D�D�D�D�D�D�D�D�D�D�D{DhD_D[DbDo���������������������{ŇŔŠŦūšŠœŇ�{�r�n�m�l�k�l�n�r�{�	��"�.�;�G�T�U�G�B�;�.�"��	������� �	��������������������������������ìù��������ùõìÞÓÇ��|�~ÇÊÓàì�4�@�M�U�`�a�S�@�'�����ݻ߻����'�4���ûлջܻллĻû����������������������C�G�O�\�h�i�h�\�[�O�C�;�6�3�6�<�C�C�C�C�4�@�M�f�r������������r�f�Y�M�@�:�3�0�4 - C R , S L 4 @ C g V 6 Z W J 9 3 l 2 H d 7 Z O M m F  " O R 7 5 8 * t s q . A � ( : N N L : t  � ! R @ R " V E B J 7  �  �  �     .    �  �  "    .  f  o  n  7  a  �  �  _  \  f  >  �  �  ]  r  d    �  �  �  �  (  �  
  �  �  �  Z  4  ?  r    �  �     0  7  #  l  :  m  u  |  U  �  �  �  �  �<T��<�C�;o<�o<49X=\)<o<u<�`B;�`B=o<u<���<T��<D��=��<�1<ě�=�C�<e`B<�<���<�9X<�<�1<�t�=��P<�`B=Y�=t�=]/<��=�P<�h=}�=#�
=+=��=,1=�^5=0 �=Ƨ�=�+=D��=T��=T��=�9X=���=��P=ix�>���=y�#=���=��P=�O�=�G�=�/=��=��=�`BB B	�B&��B�3B�{B�
B)�QB�]B�[B�B�B��B��Ba?B}iBj�BS�B��Bv�B��B!�6B	�B��B�6B:(B��B��B�`B�JBtB��B%=�B��B	%XB��B
�#B0�B:�B�dBjABkWB��BUB	��B�VB,�lA�c�B�QB��B0�B kB�TB|�B�(B�B";3B�MB �8B�B �B�QB	2�B'9�B��B��B�%B)��B��B��B�B9B%B:�BNnB�!B�=B?�B��B�B�kB!��B@BC�B��B@B�B8�B�,B(=B?;B�tB%AmBA�B	=%B� B
�7B?�B;�BB�B>�B��BFTB6�B	�uB?�B,öA���B@RB�/B��B?7B��B�DB�jB�yB"<uB��B �JB�BB9�Af��A�ۗ@C�qA�ۘ>���@�PA�W�A��B�=AIdIA�w�A.�A���BӷA��\@��?���A�A��@��AG A�mA� �A��,?��A���AA�}A��A��bAǾ@V�A)M(A�1�A��Af�"Aŗ�C�i�A�?�C�;�@m&A���A�)A�6.A*A�/
B�|A<�	@lvC��9A3DA�|6A_[�A0�7A��d@�M@�͎B(�@�:�Af��A��2@�C�xA�rW>�*{@��A�o�A�[�B�AIGKA�A-2AЈ�B�)A���@���?��A���A�\�@��AF�ZA�}�A°�Á�?�G�A�y�A�t�A҃A֏YA��DA��@�ֳA)#A��zA�wFAe�A�m^C�h#A�?rC�(�@uA��AڈAć/A��A�}bB��A=�@k�C�ׅA3�A�}aA]GA06+A��@��y@��yB �Z@���                  #                              U         :                  	      9   
   #      !   
         "         	      ;      <      
         .   %         �               /   +         .                  #         -      +               3         /                        %            +            -               +      '                  -         #                  !                                    )      +               #                                             %                           )      %                  -                                    O�>#N��UN�&N��tN�IO�N���NS�BP�N�<P��N?6O�`mNR�N��O덤N�5dNX1�O�~�N+bN���N屩NT��N� CN:��N�BO�h�N�fN���N���O�#�N���N�R�N�\,O�p�OU��N3�sN|�iNB�%P�xN
Y�O�n�OLlN�OaXPN^EJO@�YP	N��N�9,Op\N-��O
�NO�NFXcO�'Oo�@N���Nn�O�o�  `  �  �  �  p  �  �  Y  d  �    t  �    �  �  �    �  D    �  V  �  �  �  �  �  1  {  _  j  �  z  �  M  �    a  �  �  A  %  �  1  �  	  A  �  w  �  v  �  �    	�  �  �  	  绣�
;��
�D��;o$�  <�o:�o;��
;ě�;��
;ě�<t�<o<t�<t�=,1<49X<D��<�<49X<�o<e`B<T��<�C�<T��<e`B=C�<���=t�<�9X<�`B<ě�<�j<���=#�
<�<�h<�<��='�=��=49X=,1=�w=#�
=49X=T��=@�=L��=P�`>	7L=e`B=q��=ix�=�o=��=���=�7L=�7L=�\)$$&")5N[guolrpiNB5)$RPP[bgqtwytg[[RRRRRR;7<>=<7<IUV``UMIA<;;++/2<HLUXUUKH<4/++++���	 �����������������������������������������������������������������������������
 " ".,��������������������������fhr���������������wf
!









b_aht|�����������thb/565+05BCJFB:5//////������������������������)5ISTPJB5)��JGJOY[hktutphd[OJJJJotx�����������ztoooo�������
$(/.'#����`\adknnzynfa````````��������������������#/5<==<<8/#`[anz��zna``````````���������������������������������������� )6BDDB6.)           
)6>CEDA=6)@BDNN[ghnmkg^[NJDB@@.//6<HKUWUUHB<8/....���������������������������
(3<HM<#
��#04710.#�������the`ejqtw����SNU[gty{tsg[SSSSSSSS
)5BLRMKB5)|z{wvvx������������|�����������������������

���������������������������������"2691)������������

����������)BEE>96)�����
$),5?A95)Z[gt����tg][ZZZZZZZZ�����������������������������������ZUYamz��������zmfaZ ��)25BNSWWOB7) ��������������������
%)069ACB6)���������

����@DHU^afaUH@@@@@@@@@@�����������������#)/38;=9/#���������������������������������������~wz����������������~����������������������	


 ��������������������������.�;�G�`�t�|�v�u�{�y�m�`�T�G�;�'�"�#�(�.�������������������������������������������������������������������������}��E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�ù��������ùìæìóùùùùùùùùùù�Ϲܹ�������������ܹϹɹù����ùʹϼ��������������������������������������������������������������������������������[�j�q�t�g�g�o�n�g�[�B�5�������5�B�[��������������������������������������򾱾��ʾ̾˾ž�����������p�l�t�j�g�����������	����������������������������������%�4�(�������ݽĽ������Ľڽ����������������������������ý�������������I�V�b�l�i�b�V�I�F�G�I�I�I�I�I�I�I�I�I�I������%�(�+�(�������ܿο¿��ʿ���������������¼��������������������������L�U�Y�]�e�j�e�_�Y�W�T�L�@�<�@�B�L�L�L�L���(�A�H�T�]�\�S�A�5�������������a�n�w�zÇÇÇ�z�n�d�a�a�a�a�a�a�a�a�a�a�����������������������������������������s����������������������s�j�o�s�s�s�s�/�<�D�@�>�<�/�'�%�,�/�/�/�/�/�/�/�/�/�/�#�/�<�H�J�H�H�?�<�/�%�#�"�"�#�#�#�#�#�#ìù����������ýùöìäìììììììì�'�3�,�*�'������'�'�'�'�'�'�'�'�'�'�����	��#�1�6�6�/�"��	�����������������U�Z�b�n�n�q�n�n�b�U�I�B�<�9�8�<�I�P�U�U��������	�����������������������������6�@�A�=�9�6�)������)�,�6�6�6�6�6�6��"�/�H�S�[�a�`�T�;�/�#��	���������
������ ����������ټ޼������F�:�9�6�-�!��!�,�-�:�F�S�_�`�_�[�S�H�F�Ľнݽ���ݽнĽĽ����ĽĽĽĽĽĽĽ��	��"�/�:�D�D�@�8�/�"�	���������������	���������������������������s�g�Z�Y�W�g���;�G�T�`�m�t�m�`�T�G�;�5�;�;�;�;�;�;�;�;�U�a�l�n�o�n�h�a�]�U�H�<�6�<�H�S�U�U�U�UE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��0�<�K�U�W�V�N�<�#�������������������0E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E������Ǻֺպκκ˺������~�u�`�]�c�j�r�~���tāčĒĚğĠĚĕčā�t�h�[�Z�X�X�_�o�t�h�h�k�s�p�h�[�U�P�R�[�g�h�h�h�h�h�h�h�h�H�U�a�j�i�b�a�a�f�a�U�J�H�<�2�/�-�0�<�H�y���������������y�u�s�r�y�y�y�y�y�y�y�yčĚĦĭĳĴĳİħĦĚčāā�x�w�{āćčƧ�������2�=�B�J�H�0��������ƳƖƠƧ�A�M�Z�f�f�g�f�`�Z�O�M�L�A�4�.�-�4�5�A�A�-�:�F�J�:�-�(�!��������������#�-D{D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DrDqDwD{���������������������{ŇŔŠţũŠşŔőŇ�{�n�n�m�m�n�t�{�{�	��"�.�;�G�T�U�G�B�;�.�"��	������� �	��������������������������������ìùý������ùðìà×ÓÇÁÂÇÑÓàì�'�4�@�I�O�W�X�M�@�'������������'�ûлѻڻлϻû������������������ûûû��C�G�O�\�h�i�h�\�[�O�C�;�6�3�6�<�C�C�C�C�4�@�f�r�������������r�f�Y�M�@�;�5�3�3�4 - E R " S & 4 8 H g T * Z W J $ 9 i  H C 2 ^ D M m 4  ( O K 5 5 8  q s q . ? � $ 6 N N O + t  �  R 5 R " R - ? J 3  �  �  �  �  .  V  �  l  �      7  o  n  7    �  �  �  \  �  �  �  �  ]  r  A    �  �  4  �  (  �  �  /  �  �  Z  �  ?    �  �  �  �  �  7    l  �  m  0  |  U  <  �  �  �  �  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  B8  `  T  >  <  5  *      �  �  �  }  Z  >  &    �  �  s  M  -  M  d  w  �  �  �  �  �  �  �  �  �    M    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      l  x  �  �  �  {  g  O  7    �  �  �  m  )  �  c  �  w   �  p  l  k  q  r  o  k  f  b  ^  Y  T  O  H  B  A  A  B  B  C  x  �  �    N  w  �  �  �  �  �  �  |  Y  (  �  �    s  �  �  �  �  �  �  �  �  �  �  t  h  \  O  B  1           �  7  @  H  O  V  [  ]  Z  V  S  P  K  @  )    �  �  d    �  F  _  a  U  J  ?  *    �  �  �  |  O    �  �  �  n  $   �  �  �  �  �  �  �  �  �  �  �  �  t  c  R  B  1  !     �   �        �  �  �  �  �  �               �  �  m  �   �  k  n  r  t  s  s  q  n  l  m  n  p  s  w  {  �  �  �  �  +  �  �    p  a  T  G  :  *    �  �  �  �  ]  )  �  �  �  �                   �  �  �  �  �  �  �  �  �  w  i  Z  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  e  �  �  &  X  �  �  �  �  �  x  V  (  �  �  �  O  y  q  q  �  �  �  �  �  �  �  �  �  �  y  `  G  *    �  �  �  F  �  �  �  �    �    9  H  P  N  C  1  #      �  �  �  �  �  G  i  x  �  �  �  �  �  �  �  �  k  A    �  J  �  /  z  ,  D  6  (      �  �  �  �  �  �  �  �  �  |  q  f  [  P  D  �  �  �  �  �      �  �  �  �  �  �  l  M  +  
  �  �  v  �  �  �  �  �  �  �  �  |  [  5    �  �  �  a  Y  <    �  >  K  V  V  V  S  O  E  9  )  Q  �  �  �  �  �  �  �  �  �  �    :  e  �  �  �  �  ~  i  R  8    �  �  �  |  K    �  �  �  �  �  �  �  d  G  &    �  �  �  e  9     �   �   z   I  �  �  �  �  �  �  �  �  �  �  �  q  Z  C  ,    �  �  w  I    A  j  �  �  �  �  �  y  U  /    �  �  C  �  K  �  �  A  �  �  �  �  �  q  Y  F  4  !    �  �  �  �  �  �  v  J    �  �    U    �  �  �    .  +    �  �  d    �  T  �  �  {  `  D  '  
  �  �  �  m  B    �  �  �  ~  \  5  �  ~    W  [  ^  _  ^  [  T  G  1      �  �  �  �  N    �  u  �  `  d  g  h  j  h  e  ]  T  A  (    �  �  �  c  B     �   �  �  j  �  f  a  X  S  N  E  ?  7  -    
  �  �  �  �  )   �  z  v  s  o  l  h  d  `  \  X  S  L  E  >  7  !     �   �   �  �  �  �  �  �  �  �  �  �  �  �  |  y  r  b  ?    �    v  9  E  K  C  7  (      �  �  �  �  �  �  �  �  �  y  :  �  �  �  �  �  y  n  c  X  M  C  8  -  "       �   �   �   �   �    m  [  C  (    �  �  �  �  �  �  {  Z  8    �  �  �  R  a  Q  A  0      �  �  �  �  �  I  �  �  �  T    �  �  b  J  x  �  �  �  n  Q  .    �  �  {  >  �  �  '  �      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  l  V  A  +     9  A  3    �  �  �  ~  P  "  �  �  r  #  �  7  �  �   �      $            �  �  �  v  D    �  ~  $  �    �  �  �  �  �  �  �  x  _  F  ,    �  �  �  }  ^  A  *    �  �  1    �  �  �  �  �  �  �  �  �  �  b  /  �  �  �  �  \  4  �  �  �  �  �  �  �  �  �  �  �  �  �  `    �    ?   �   �  z  �  �  	  	  	  �  �  �  p  (  �  q    �  �     �  L   �  A  >  2  &    �  �  �  |  ?  �  �  �  /  �  �  )  �  :  "  �  �  �  �  �  �  �  �  �  �  �  b  7    �  :  �  �  �   �  w  Y  ;    �  �  �  �  i  G  -      �  �  �  Y  @  (    �  �  T    �  )  �  �  �  �  K  �    �  �  �  �  P  
�  �  v  q  l  g  a  U  J  >  1  "      �  �  �  �  �  p  `  O  �  �  �  �  �  t  W  9    �  �  �  [    �  |  
  �    m  �  �  �  �  �  �  �  i  P  2    �  �  �  t  D    �  A          �  �  �  �  �  �  �  �  �  �  �  �  �  p  X  @  (  	i  	�  	�  	�  	�  	�  	�  	T  	  �  s    �  )  �    _  }  �  �  P  d  �  �  �  �  �  �  x  Q    �  �  5  �  `  �  E  E  8  �  �  �  �  �  �  �  }  ]  8    �  �  �  V    �  T  �    	        �  �  �  �  �  �  �  �  �  |  h  U  >    �  �  �  �  �  �  �  �  ^  6  
  �  |    �  .  �  8  �  �  �  d